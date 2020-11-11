import random

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def evaluate(df):
    y_test = df.observation.to_numpy()
    y_pred = df.prediction.to_numpy()
    num_values = len(y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    num_pos_preds = accuracy * num_values
    num_neg_preds = num_values - num_pos_preds
    FN = num_pos_preds * (1 - recall)
    TP = num_pos_preds - FN
    TN = num_pos_preds - TP
    FP = num_neg_preds - TN
    # print("Number of test items: " + str(num_values))
    # print("TP: " + str(TP))
    # print("TN: " + str(TN))
    # print("FP: " + str(FP))
    # print("FN: " + str(FN))
    # print("Precision: " + str(precision))
    # print("Recall: " + str(recall))
    print("Accuracy: " + str((TP + TN) / (TP + TN + FP + FN)))
    print("Classification Error: " + str((FP + FN) / (TP + TN + FP + FN)))
    # print("Positive Predictive Value: " + str(TP / (TP + FP)))
    # print("Demographic Parity: " + str((TP + FP) / (TP + TN + FP + FN)))
    # print("False Positive Rate: " + str(FP / (TN + FP)))


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


df = pd.read_csv("data/Food_Inspections.csv")
indexNames = df[(df.Results == 'Out of Business')].index
df.drop(indexNames, inplace=True)
indexNames = df[(df.Results == 'Not Ready')].index
df.drop(indexNames, inplace=True)
indexNames = df[(df.Results == 'Pass w/ Conditions')].index
df.drop(indexNames, inplace=True)
indexNames = df[(df.Results == 'Business Not Located')].index
df.drop(indexNames, inplace=True)
indexNames = df[(df.Results == 'No Entry')].index
df.drop(indexNames, inplace=True)
indexNames = df[(df['Inspection Type'] != 'Complaint')].index
df.drop(indexNames, inplace=True)
indexNames = df[(df.Results.isnull())].index
df.drop(indexNames, inplace=True)
indexNames = df[(df.Latitude.isnull())].index
df.drop(indexNames, inplace=True)
indexNames = df[(df.Longitude.isnull())].index
df.drop(indexNames, inplace=True)
indexNames = df[(df['Inspection Type'].isnull())].index
df.drop(indexNames, inplace=True)
indexNames = df[(df['Risk'].isnull())].index
df.drop(indexNames, inplace=True)
indexNames = df[(df['Zip'].isnull())].index
df.drop(indexNames, inplace=True)
indexNames = df[(df['Facility Type'].isnull())].index
df.drop(indexNames, inplace=True)
df.drop("Inspection ID", axis=1, inplace=True)
df.drop("DBA Name", axis=1, inplace=True)
df.drop("AKA Name", axis=1, inplace=True)
df.drop("License #", axis=1, inplace=True)
df.drop("Address", axis=1, inplace=True)
df.drop("City", axis=1, inplace=True)
df.drop("State", axis=1, inplace=True)
df.drop("Inspection Date", axis=1, inplace=True)
df.drop("Violations", axis=1, inplace=True)
df.drop("Location", axis=1, inplace=True)
df.drop("Inspection Type", axis=1, inplace=True)
facility_type_list = list(set(df['Facility Type']))
df.loc[df['Results'] == "Pass", 'Results'] = 1
df.loc[df['Results'] == "Fail", 'Results'] = 0
df.loc[df['Risk'] == "Risk 1 (High)", 'Risk'] = 1
df.loc[df['Risk'] == "All", 'Risk'] = 1
df.loc[df['Risk'] == "Risk 2 (Medium)", 'Risk'] = 2
df.loc[df['Risk'] == "Risk 3 (Low)", 'Risk'] = 3

for i in range(len(facility_type_list)):
    df.loc[df['Facility Type'] == facility_type_list[i], 'Facility Type'] = i

df.to_csv("data/food_inspection_proof.csv", sep=',', encoding='utf-8', index=False, header=True)
x = df.drop('Results', axis=1)
y = df.Results
y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

minmaxscale = MinMaxScaler()
scaler = minmaxscale.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
clf = LogisticRegression()
clf.fit(x_train_scaled, y_train)
y_pred = clf.predict(x_test_scaled)

x_plot_correct = []
y_plot_correct = []
x_plot_incorrect = []
y_plot_incorrect = []

for i in range(len(y_test)):
    if y_test[i] == 1:
        x_plot_correct.append(x_test_scaled[i][3])
        y_plot_correct.append(x_test_scaled[i][4])
    else:
        x_plot_incorrect.append(x_test_scaled[i][3])
        y_plot_incorrect.append(x_test_scaled[i][4])

step = 0.1
grid_correct = {}
key = 0
for pivot_x in np.arange(0.0, 1.0, step):
    for pivot_y in np.arange(0.0, 1.0, step):
        values = []
        key += 1
        for i in range(len(x_plot_correct)):
            if pivot_x <= x_plot_correct[i] <= pivot_x + step and pivot_y <= y_plot_correct[i] <= pivot_y + step:
                value = [x_plot_correct[i], y_plot_correct[i]]
                values.append(value)
        grid_correct.update({key: values})

grid_incorrect = {}
key = 0
for pivot_x in np.arange(0.0, 1.0, step):
    for pivot_y in np.arange(0.0, 1.0, step):
        values = []
        key += 1
        for i in range(len(x_plot_incorrect)):
            if pivot_x <= x_plot_incorrect[i] <= pivot_x + step and pivot_y <= y_plot_incorrect[i] <= pivot_y + step:
                value = [x_plot_incorrect[i], y_plot_incorrect[i]]
                values.append(value)
        grid_incorrect.update({key: values})

grid_entropy = {}
entropy_list = []
min_grid_features = []
min_grid_observations = []
max_grid_features = []
max_grid_observations = []
min_flag = True
max_flag = True
for key in grid_correct:
    entropy1 = 0
    if len(grid_incorrect[key]) != 0:
        p = len(grid_correct[key]) / (len(grid_correct[key]) + len(grid_incorrect[key]))
        if p != 0:
            entropy1 = -p * np.log2(p)
    entropy2 = 0
    if len(grid_correct[key]) != 0:
        p = len(grid_incorrect[key]) / (len(grid_correct[key]) + len(grid_incorrect[key]))
        if p != 0:
            entropy2 = -p * np.log2(p)
    entropy = entropy1 + entropy2
    grid_entropy.update({key: entropy})
    entropy_list.append(entropy)

rho = 0.1
k = 100
for key in grid_correct:
    sample1 = grid_correct.get(key)
    sample2 = grid_incorrect.get(key)
    sample = sample1 + sample2
    sample_features = []
    sample_observation = []

    numCovered = 0
    numUncovered = 0
    for item in sample:
        for i in range(len(x_test_scaled)):
            if item[0] == x_test_scaled[i][3] and item[1] == x_test_scaled[i][4]:
                sample_features.append(x_test_scaled[i])
                sample_observation.append(y_test[i])
                neighbourCount = 0
                for j in range(len(x_train_scaled)):
                    a = (item[0], item[1])
                    b = (x_train_scaled[j][3], x_train_scaled[j][4])
                    if distance.euclidean(a, b) <= rho:
                        neighbourCount += 1
                        if neighbourCount >= k:
                            break
                if neighbourCount >= k:
                    numCovered += 1
                else:
                    numUncovered += 1

    if len(sample_features) != 0:
        # print("Entropy:" + str(grid_entropy.get(key)))
        # print("Sample is empty!")
        # print("-------------------------------")
        # else:
        if 0.5 < grid_entropy.get(key) < 0.65 and (
                len(grid_correct[key]) + len(grid_incorrect[key])) > 25 and min_flag == True:
            min_grid_features.extend(sample_features)
            min_grid_observations.extend(sample_observation)
            min_flag = False
            # print("Min Entropy:")
            # print(grid_entropy.get(key))

        if grid_entropy.get(key) >= 0.99 and (
                len(grid_correct[key]) + len(grid_incorrect[key])) > 25 and max_flag == True:
            max_grid_features.extend(sample_features)
            max_grid_observations.extend(sample_observation)
            max_flag = False
            # print("Max Entropy:")
            # print(grid_entropy.get(key))

        print("# of True-> " + str(len(grid_correct[key])) + ":" + str(len(grid_incorrect[key])) + " <-# of False")
        print("Entropy:" + str(grid_entropy.get(key)))
        sample_prediction = clf.predict(pd.DataFrame(sample_features))
        results = pd.DataFrame(list(zip(sample_prediction, sample_observation)), columns=['prediction', 'observation'])
        evaluate(results)
        print("# of Covered: " + str(numCovered))
        print("# of Uncovered: " + str(numUncovered))
        print("-------------------------------")

x_axis = []
y_axis = []

for pivot_x in np.arange(0.0, 1.0, step):
    x_axis.append(np.round(pivot_x, 2))

for pivot_y in np.arange(0.0, 1.0, step):
    y_axis.append(np.round(pivot_y, 2))

B = np.reshape(entropy_list, (len(y_axis), len(x_axis)))
entropy_ = B.T

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18, 5))
# ax[0].plot(x_plot_correct, y_plot_correct, 'bo', markersize=2)
# ax[0].plot(x_plot_incorrect, y_plot_incorrect, 'ro', markersize=2)
im, cbar = heatmap(entropy_, y_axis, x_axis, ax=ax[1], cbarlabel="Entropy", cmap="viridis")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
fig.tight_layout()
plt.xlabel('Latitude')
plt.ylabel('Longitude')

print(min_grid_features)
print(min_grid_observations)
print(max_grid_features)
print(max_grid_observations)

size = 8
train_size = []
accuracy_list = []
temp = list(zip(max_grid_features, max_grid_observations))
random.shuffle(temp)
res1, res2 = zip(*temp)
for f in res1:
    x = res1[:size]
    y = res2[:size]
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    if len(x_train) not in train_size:
        train_size.append(len(x_train))
        accuracy_list.append(accuracy)
        print("train size:" + str(len(x_train)))
        print("accuracy:" + str(accuracy))
    size += 1
ax[0].plot(train_size, accuracy_list, 'xb-', label="High Entropy")
for i in range(10):
    try:
        size = 8
        train_size = []
        accuracy_list = []
        temp = list(zip(min_grid_features, min_grid_observations))
        random.shuffle(temp)
        res3, res4 = zip(*temp)
        for f in res3:
            x = res3[:size]
            y = res4[:size]
            print(y)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
            clf = LogisticRegression()
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            if len(x_train) not in train_size:
                train_size.append(len(x_train))
                accuracy_list.append(accuracy)
                print("train size:" + str(len(x_train)))
                print("accuracy:" + str(accuracy))
            size += 1
        ax[0].plot(train_size, accuracy_list, '.r-', label="Low Entropy")
        ax[0].legend(loc='best', numpoints=1, ncol=2, fontsize=8)
    except:
        continue
    plt.show()


