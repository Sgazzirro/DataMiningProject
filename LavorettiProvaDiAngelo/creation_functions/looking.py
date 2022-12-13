import matplotlib.pyplot as plt
import seaborn as sns
import random
from dataset_creation import create_time_series_speed_up


def have_a_look_at(activity_code, TRIAL_CODES, signal,num_actors=4):
    """
    activity_code: the activity we want to have a look at
    return: 6 plots (2x3), plotting users accelerations at 12-subjects groups
    """
    ACTIVITY_CODES = ["dws", "jog", "sit", "std", "ups", "wlk"]

    if num_actors % 2 != 0:
        num_actors += 1

    datasets = []
    actors = random.sample(range(0, 25), num_actors)
    label = ["subject_" + str(actor) for actor in actors]
    for i in actors:
        datasets.append(create_time_series_speed_up(activity_code, TRIAL_CODES, [i], mode="raw"))

    for i in range(0, len(datasets)):
        plt.plot(datasets[i].index, datasets[i][signal])
    plt.legend(label)


