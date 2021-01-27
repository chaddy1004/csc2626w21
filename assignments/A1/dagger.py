import argparse
import os
import random
import time

import imageio
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from full_state_car_racing_env import FullStateCarRacingEnv
from train_policy import train_discrete
from utils import DEVICE, str2bool

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

        # getting the cross_track_error
        # absolute value was used since errors had both positive and negative values
        # for comparison, I decided that just getting the magnitude of TOTAL error is enough for comparison
        cross_track_error_heading += abs(env.get_cross_track_error(env.car, env.track)[0])  # get the error_heading
        cross_track_error_dist += abs(env.get_cross_track_error(env.car, env.track)[1])  # get the error_dist

        if done:
            print(t)
            duration = t
            break

        # not changed from original code
        expert_steer = expert_action[0]  # [-1, 1]
        expert_gas = expert_action[1]  # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]

        # steering action is the prediction from the policy network
        learner_action[0] = steering_network.eval(state, device=DEVICE)
        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        # save the current state (image) with the expert's action. This is the data aggregation part
        imageio.imsave(os.path.join(args.train_dir, 'expert_%d_%d_%f.jpg' % (run_id, t, expert_steer)),
                       state)
    env.close()

    # normalize the amount of error by dividing by duration it ran for
    cross_track_error_dist /= duration
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

    # The first weight training
    print('TRAINING LEARNER ON INITIAL DATASET')

    # train for n_epoch times (20 as default; engineering decision I made)
    for i in range(args.n_epochs):
        train_discrete(model=driving_policy, iterator=training_iterator, opt=opt, args=args)
        weight_name = f"./weights/dagger_{0}_learner_0.weights"
        torch.save(driving_policy.state_dict(), weight_name)

    durations = []
    cross_track_errors_dist = []
    cross_track_errors_heading = []
    for i in range(1, args.dagger_iterations):
        # aggregating new data by running the current policy on simulation
        print('GETTING EXPERT DEMONSTRATIONS')
        duration, cross_track_error_dist, cross_track_error_heading = run(steering_network=driving_policy, run_id=i,
                                                                          args=args)
        # new training dataset whenever there is new data from previous run
        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        training_dataset = DrivingDataset(root_dir=args.train_dir,
                                          categorical=True,
                                          classes=args.n_steering_classes,
                                          transform=data_transform)

        # must re-make training dataloader since the dataset is now updated with aggregation of new data
        training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)

        # new weight initialized per new dagger iteration
        driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
        # new optimzer initialization with the new policy's weight
        opt = torch.optim.Adam(params=driving_policy.parameters(), lr=args.lr)
        # train for n_epochs with the aggregated new dataset
        for _ in range(args.n_epochs):
            train_discrete(model=driving_policy, iterator=training_iterator, opt=opt, args=args)

        # this line commented out since performing validation on each dagger iteration just took so uch time
        # and technically this does not make a difference in the final result
        # avg_acc = test_discrete(model=driving_policy, iterator=validation_iterator, args=args, opt=None)

        # save current weight
        weight_name = f"./weights/dagger_{i}_learner_0.weights"
        torch.save(driving_policy.state_dict(), weight_name)

        # append duration, distance cross track errors, and heading cross track errors
        durations.append(duration)
        cross_track_errors_dist.append(cross_track_error_dist)
        cross_track_errors_heading.append(cross_track_error_heading)

    print(f"Durations of each run: {durations}")
    print(f"C.T.E. dist of each run: {cross_track_errors_dist}")
    print(f"C.T.E. heading of each run: {cross_track_errors_heading}")

    x_axis = [i for i in range(len(cross_track_errors_dist))]

    fig, axes = plt.subplots(3, 1, sharex='col')
    fig.suptitle('DAgger Results')

    d = {"DAgger Iteration": x_axis, "Dist Error": cross_track_errors_dist, "Head Error": cross_track_errors_heading,
         "Simulation Duration": durations}
    experiment = pd.DataFrame(data=d)
    print(experiment)

    cte_plot = sns.lineplot(ax=axes[0], data=experiment, x="DAgger Iteration", y="Dist Error")
    cte_plot.set_title("Dist Error vs DAgger Iteration")
    cte_plot.figure.savefig("DistErrorPlot.png")

    head_plot = sns.lineplot(ax=axes[1], data=experiment, x="DAgger Iteration", y="Head Error")
    head_plot.set_title("Heading Error vs DAgger Iteration")
    head_plot.figure.savefig("HeadingErrorPlot.png")

    duration_plot = sns.lineplot(ax=axes[2], data=experiment, x="DAgger Iteration", y="Simulation Duration")
    duration_plot.set_title("Simulation Duration vs DAgger Iteration")
    duration_plot.figure.savefig("dagger_iterations.png")

    fig.savefig("dagger_iterations.png")
