import argparse
import os
import time

import scipy.misc
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from full_state_car_racing_env import FullStateCarRacingEnv

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from train_policy import train_discrete, test_discrete
from utils import DEVICE, str2bool
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
import random

sns.set_theme(style="darkgrid")

torch.manual_seed(2626)
np.random.seed(19940513)
random.seed(a=19971124)


def run(steering_network, run_id, args):
    env = FullStateCarRacingEnv()
    env.reset()

    learner_action = np.array([0.0, 0.0, 0.0])
    timesteps = 100000
    duration = 0
    cross_track_error_heading = 0
    cross_track_error_dist = 0
    for t in range(timesteps):
        env.render()

        state, expert_action, reward, done, _ = env.step(learner_action)
        cross_track_error_heading += abs(env.get_cross_track_error(env.car, env.track)[0])  # get the error_heading
        cross_track_error_dist += abs(env.get_cross_track_error(env.car, env.track)[1])  # get the error_dist
        if done:
            print(t)
            duration = t
            break

        expert_steer = expert_action[0]  # [-1, 1]
        expert_gas = expert_action[1]  # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]

        learner_action[0] = steering_network.eval(state, device=DEVICE)
        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        imageio.imsave(os.path.join(args.train_dir, 'expert_%d_%d_%f.jpg' % (run_id, t, expert_steer)),
                       state)
    env.close()
    cross_track_error_dist /= duration  # normalize the amount of error by dividing by duration it ran for
    cross_track_error_heading /= duration
    return duration, cross_track_error_dist, cross_track_error_heading


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset_1/train_dagger')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset_1/val')
    parser.add_argument("--weights_out_file",
                        help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="", default=10)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=False)

    args = parser.parse_args()

    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                         transforms.RandomRotation(degrees=80),
                                         transforms.ToTensor()])

    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)

    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)

    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)

    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    args.start_time = time.time()

    print(driving_policy)
    print(opt)
    print(args)

    best_val_accuracy = 0

    opt = torch.optim.Adam(params=driving_policy.parameters(), lr=args.lr)

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####

    print('TRAINING LEARNER ON INITIAL DATASET')
    for i in range(20):
        train_discrete(model=driving_policy, iterator=training_iterator, opt=opt, args=args)
    # test_discrete(model=driving_policy, iterator=validation_iterator, args=args, opt=None)

    print('GETTING EXPERT DEMONSTRATIONS')
    durations = []
    cross_track_errors_dist = []
    cross_track_errors_heading = []
    for i in range(args.dagger_iterations):
        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        duration, cross_track_error_dist, cross_track_error_heading = run(steering_network=driving_policy, run_id=i,
                                                                          args=args)
        # new training dataset whenever there is new data from previous run
        training_dataset = DrivingDataset(root_dir=args.train_dir,
                                          categorical=True,
                                          classes=args.n_steering_classes,
                                          transform=data_transform)

        training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
        driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
        opt = torch.optim.Adam(params=driving_policy.parameters(), lr=args.lr)
        for _ in range(20):
            train_discrete(model=driving_policy, iterator=training_iterator, opt=opt, args=args)
        weight_name = f"./weights/dagger_{i}_learner_0.weights"
        torch.save(driving_policy.state_dict(), weight_name)
        durations.append(duration)
        cross_track_errors_dist.append(cross_track_error_dist)
        cross_track_errors_heading.append(cross_track_error_heading)

    print(f"Durations of each run: {durations}")
    print(f"C.T.E. dist of each run: {cross_track_errors_dist}")
    print(f"C.T.E. heading of each run: {cross_track_errors_heading}")

    x_axis = [i for i in range(len(cross_track_errors_dist))]
    # x_axis_duration = [i for i in range(len(duration))]

    d = {"DAgger Iteration": x_axis, "Dist Error": cross_track_errors_dist, "Head Error": cross_track_errors_heading,
         "Simulation Duration": durations}
    experiment = pd.DataFrame(data=d)
    print(experiment)

    cte_plot = sns.lineplot(data=experiment, x="DAgger Iteration", y="Dist Error")
    cte_plot.set_title("Dist Error vs DAgger Iteration")
    cte_plot.figure.savefig("DistErrorPlot.png")
    plt.clf()

    head_plot = sns.lineplot(data=experiment, x="DAgger Iteration", y="Head Error")
    head_plot.set_title("Heading Error vs DAgger Iteration")
    head_plot.figure.savefig("HeadingErrorPlot.png")
    plt.clf()

    duration_plot = sns.lineplot(data=experiment, x="DAgger Iteration", y="Simulation Duration")
    duration_plot.set_title("Simulation Duration vs DAgger Iteration")
    duration_plot.figure.savefig("DurationPlot.png")
    plt.clf()
