"""
file: HW_05_KALLURWAR_Anurag.py
description: This program
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""


import warnings
import sys
import os
import math

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONSTANTS
NUM_OF_VALUES = 100
MIN_ANGLE = 0
MAX_ANGLE = 90
OFFSET_ANGLE = 15
TARGET = 'TripType'
FEATURE_NAME = 'projections'


def clean_data(df: pd.DataFrame):
    """
    Cleaning the dataframe
    :param df: input dataframe
    :return: cleaned dataframe
    """
    return df.dropna()


def read_file(file_name: str):
    """
    Read the CSV file and return dataframe
    :param thread_index: Index of thread
    :param file_paths: filename
    :return: dataframe
    """
    print("Reading file: " + file_name)
    # Skipping first line containing "HEADER" string
    dataframe = pd.read_csv(file_name, skiprows=[0], low_memory=False)
    dataframe = clean_data(dataframe)
    return dataframe


def find_classifier_threshold(projections_df: pd.DataFrame):
    """
    Find the best threshold for classification along the projection vector.
    :param projections_df: input dataframe containing the projection values
    along the projection vector
    :param feature_name: feature, whose threshold is being calulated
    :return: best threshold, minimum number of mistakes
    """
    # print("Searching the best threshold for these projections...")
    # Threshold range
    min_value = projections_df[FEATURE_NAME].min()
    max_value = projections_df[FEATURE_NAME].max()
    # print(min_value, max_value)
    # Offset
    delta_value = (max_value - min_value) / NUM_OF_VALUES
    # Initializations
    min_mistakes = np.inf
    best_threshold = 0
    false_positive_population = 0
    false_negative_population = 0
    threshold = min_value - delta_value
    # Searching the best threshold for classification
    while threshold <= (max_value - delta_value):
        # the number of bad trips (class != 1) who are <= threshold
        false_positive_population = projections_df[(projections_df[
                                                        FEATURE_NAME] <=
                                                    threshold) & (
                projections_df[TARGET] != 'Good')].shape[0]
        # the number of good trips (class == 1) who are > threshold
        false_negative_population = projections_df[(projections_df[
                                                        FEATURE_NAME] >
                                                    threshold) & (
                projections_df[TARGET] == 'Good')].shape[0]
        # Total mistakes
        mistakes = false_positive_population + false_negative_population
        # print(threshold, mistakes)
        # Minimizing the mistakes and selecting best threshold
        if mistakes < min_mistakes:
            min_mistakes = mistakes
            best_threshold = threshold
        threshold += delta_value
    return best_threshold, min_mistakes


def calculate_projections_and_threshold(angle: float, projections_df:
pd.DataFrame):
    """
    Calculate the projection unit vector for the angle and then calculates 
    the corresponding projection values for the data
    :param angle: 
    :param projections_df: DF with projections
    :return: 
    """
    # Calculating unit projection vector for this angle
    projection_vector = [np.cos(angle * np.pi / 180), np.sin(angle * np.pi /
                                                             180)]
    # Calculating projections
    projections_df[FEATURE_NAME] = projections_df.apply(lambda row:
                                                        np.dot(
                                                            projection_vector,
                                                            [row['RoadDist'],
                                                             row['ElevationChange']]),
                                                        axis=1)
    # Finding best threshold for the feature
    threshold, mistakes = find_classifier_threshold(projections_df)
    return threshold, mistakes, projections_df


def projection_angle_selection(trips_df: pd.DataFrame):
    """
    Selects the best feature for a one rule classifier
    :param growth_df: input dataframe
    :return: selected angle, best threshold, minimum number of mistakes,
    projection values for the data
    """
    # Initializations
    best_projections_df = trips_df
    best_angle = MIN_ANGLE
    min_mistakes = np.inf
    best_threshold = 0
    projections_df = trips_df
    angle = MIN_ANGLE
    # Finding best projection angle0 for classification
    while angle <= MAX_ANGLE:
        # Calculating best threshold, minimum mistakes, projection values for
        # the angle
        threshold, mistakes, projections_df = \
            calculate_projections_and_threshold(angle, projections_df)
        # print(threshold, mistakes)
        # Minimizing the mistakes and selecting best angle
        if mistakes <= min_mistakes:
            min_mistakes = mistakes
            best_threshold = threshold
            best_angle = angle
            best_projections_df = projections_df
        angle += OFFSET_ANGLE
    return best_angle, best_threshold, min_mistakes, best_projections_df


def gradient_descent(angle: float, step_size: float, projections_df:
pd.DataFrame, min_mistakes: float):
    """
    This method calculates the best projection angle by minimizing the 
    number of mistakes through gradient descent.
    :param angle: The angle
    :param step_size: Step Size for gradient descent
    :param projections_df: The input data
    :param min_mistakes: minimum number of mistakes
    :return: selected angle, best threshold, minimum number of mistakes, 
    projection values for the data
    """
    # Base cases: till step size 0.5 and only angles between 0 and 90
    if step_size < 0.5:
        return None, None, None, None
    if angle < 0 or angle > 90:
        return None, None, None, None
    # Calculating best threshold, minimum mistakes, projection values for the
    # angle
    threshold, mistakes, projections_df = \
        calculate_projections_and_threshold(angle, projections_df)
    # Assuming this is minimum number of mistakes
    min_mistakes = mistakes
    best_threshold = threshold
    best_angle = angle
    best_projections_df = projections_df
    # Calling function for angle - step_size
    best_angle_left, best_threshold_left, mistakes_left, \
        projections_df_left = gradient_descent(angle - step_size,
                                               step_size / 2,
                                               projections_df,
                                               min_mistakes)
    if mistakes_left is not None and mistakes_left < min_mistakes:
        min_mistakes = mistakes_left
        best_threshold = best_threshold_left
        best_angle = best_angle_left
        best_projections_df = projections_df_left
    # Calling function for angle + step_size
    best_angle_right, best_threshold_right, mistakes_right, \
        projections_df_right = gradient_descent(angle + step_size,
                                               step_size / 2,
                                                projections_df,
                                                min_mistakes)
    # Minimizing the mistakes and selecting best angle
    if mistakes_right is not None and mistakes_right < min_mistakes:
        min_mistakes = mistakes_right
        best_threshold = best_threshold_right
        best_angle = best_angle_right
        best_projections_df = projections_df_right
    return best_angle, best_threshold, min_mistakes, best_projections_df


def visualize_data(trips_df: pd.DataFrame, best_angle: float, best_threshold:
float):
    """
    Visualize the projection vector and decision boundary for the data
    :param trips_df: The input dataset
    :param best_angle: best projection angle for the projection vector
    :param best_threshold: best threshold along the projection vector
    :return: None
    """
    good_trips = trips_df[trips_df[TARGET] == 'Good']
    bad_trips = trips_df[trips_df[TARGET] == 'Bad']
    x_max = trips_df['RoadDist'].max() + 1
    y_max = trips_df['ElevationChange'].max() + 10
    projection_vector = [np.cos(best_angle * np.pi / 180), np.sin(best_angle
                                                                  * np.pi /
                                                                  180)]
    # Decision Boundary perpendicular to projection vector
    decision_boundary = [-projection_vector[1], projection_vector[0]]
    # Decision boundary line end points
    # Common point for projection vector and decision boundary
    point = [best_threshold * np.cos(best_angle * np.pi / 180),
             best_threshold * np.sin(best_angle * np.pi / 180)]
    # For y = 0
    x = ((0 - point[1]) / (-1 / np.tan(best_angle * np.pi / 180))) + (point[0])
    point1 = [x, 0]
    # For y = y_max
    x = ((y_max - point[1]) / (-1 / np.tan(best_angle * np.pi / 180))) + (
        point[0])
    point2 = [x, y_max]
    print("\nPlotting scatter plot...")
    print("Close the window to continue!")
    # Plotting
    fig, ax = plt.subplots()
    fig.suptitle("Trips Dataset")
    ax.set_title('Classification through gradient descent minimization')
    ax.scatter(good_trips['RoadDist'], good_trips['ElevationChange'], s=20,
               c='b', label='Good Trips')
    ax.scatter(bad_trips['RoadDist'], bad_trips['ElevationChange'], s=15,
               c='r', label='Bad Trips', marker='D')
    ax.plot([0, point[0]], [0, point[1]],
            linestyle='solid', color='g', label='Projection Vector')
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]],
            linestyle='dotted', color='m', label='Decision Boundary')
    ax.set_xlabel("Road Distance (miles)", fontsize=8)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Change in elevation (feet)", fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.show()


def process(trips_df: pd.DataFrame):
    """
    This function compute features and creates a one rule classifier for the
    given data by selecting the best feature
    :param trips_df: Input data
    :return: None
    """
    # Part 1.
    # Computing Best projection angle (multiple of 15deg)
    print("\n\n===== Computing Best projection angle (multiple of 15deg) =====")
    best_angle, best_threshold, min_mistakes, projections_df, \
        = projection_angle_selection(trips_df)
    print("\nResults:")
    print("Best Angle: " + str(best_angle))
    print("Best Threshold: " + str(best_threshold))
    print("Minimum mistakes: " + str(min_mistakes))

    # Gradient Descent to find the best Projection angle
    print("\n\n===== Gradient Descent to find the best Projection angle =====")
    best_angle, best_threshold, min_mistakes, best_projections_df = \
        gradient_descent(best_angle, OFFSET_ANGLE / 2, projections_df, np.inf)
    print("\nResults:")
    print("Best Angle: " + str(best_angle))
    print("Best Threshold: " + str(best_threshold))
    print("Minimum mistakes: " + str(min_mistakes))

    # Part 2.
    # Plot of the data with the best projection vector and decision boundary
    print("\n\n===== Plot of the data with the best projection vector and "
          "decision boundary =====")
    visualize_data(trips_df, best_angle, best_threshold)


def main():
    """
    The main function
    :return: None
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if len(sys.argv) < 2:
        print("Missing Argument!")
        print("Usage: HW_05_KALLURWAR_Anurag.py <filename.csv>")
        return
    file_name = sys.argv[1].strip()
    if not os.path.isfile(os.getcwd() + "\\" + file_name):
        print("Please put " + file_name + " in the execution folder!")
        return
    trips_df = read_file(file_name)
    print(trips_df)
    process(trips_df)


if __name__ == '__main__':
    main()  # Calling Main Function
