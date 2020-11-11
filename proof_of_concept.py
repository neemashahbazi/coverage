import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.evaluate import bias_variance_decomp
from scipy.spatial import distance
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import tree

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

print(df.shape)

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

# inspection_type_list = list(set(df['Inspection Type']))
facility_type_list = list(set(df['Facility Type']))

df.loc[df['Results'] == "Pass", 'Results'] = 1
df.loc[df['Results'] == "Fail", 'Results'] = 0
df.loc[df['Risk'] == "Risk 1 (High)", 'Risk'] = 1
df.loc[df['Risk'] == "All", 'Risk'] = 1
df.loc[df['Risk'] == "Risk 2 (Medium)", 'Risk'] = 2
df.loc[df['Risk'] == "Risk 3 (Low)", 'Risk'] = 3

for i in range(len(facility_type_list)):
    df.loc[df['Facility Type'] == facility_type_list[i], 'Facility Type'] = i

# for i in range(len(inspection_type_list)):
#     df.loc[df['Inspection Type'] == inspection_type_list[i], 'Inspection Type'] = i

print(df.shape)
df.to_csv("data/food_inspection_proof.csv", sep=',', encoding='utf-8', index=False, header=True)
x = df.drop('Results', axis=1)
y = df.Results
y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

minmaxscale = MinMaxScaler()
scaler = minmaxscale.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

clf = tree.DecisionTreeClassifier()
clf.fit(x_train_scaled, y_train)
weights = clf.coef_
abs_weights = np.abs(weights)
print(abs_weights)
y_pred = clf.predict(x_test_scaled)

rho = 0.02
k = 100

covered_y_pred = []
covered_y_test = []
uncovered_y_pred = []
uncovered_y_test = []

covered_x_test = []
uncovered_x_test = []

y_test = np.array(y_test)
y_train = np.array(y_train)

uncovered_x_plot_correct = []
uncovered_y_plot_correct = []
uncovered_x_plot_incorrect = []
uncovered_y_plot_incorrect = []

covered_x_plot_correct = []
covered_y_plot_correct = []
covered_x_plot_incorrect = []
covered_y_plot_incorrect = []

covered_x_plot_predicted_one = []
covered_y_plot_predicted_one = []
covered_x_plot_predicted_zero = []
covered_y_plot_predicted_zero = []

covered_x_plot_real_one = []
covered_x_plot_real_zero = []
covered_y_plot_real_one = []
covered_y_plot_real_zero = []

uncovered_x_plot_predicted_one = []
uncovered_x_plot_predicted_zero = []
uncovered_y_plot_predicted_one = []
uncovered_y_plot_predicted_zero = []

uncovered_x_plot_real_one = []
uncovered_x_plot_real_zero = []
uncovered_y_plot_real_one = []
uncovered_y_plot_real_zero = []

duration = []
for i in range(len(x_test_scaled)):
    neighbourCount = 0
    for j in range(len(x_train_scaled)):
        a = (x_test_scaled[i][2], x_test_scaled[i][4])
        b = (x_train_scaled[j][2], x_train_scaled[j][4])
        if distance.euclidean(a, b) <= rho:
            neighbourCount += 1
            if neighbourCount >= k:
                break
    if neighbourCount >= k:
        covered_y_pred.append(y_pred[i])
        covered_y_test.append(y_test[i])
        covered_x_test.append(x_test_scaled[i])
        if y_pred[i] == 1:
            covered_x_plot_predicted_one.append(x_test_scaled[i][2])
            covered_y_plot_predicted_one.append(x_test_scaled[i][4])
        else:
            covered_x_plot_predicted_zero.append(x_test_scaled[i][2])
            covered_y_plot_predicted_zero.append(x_test_scaled[i][4])
        if y_test[i] == 1:
            covered_x_plot_real_one.append(x_test_scaled[i][2])
            covered_y_plot_real_one.append(x_test_scaled[i][4])
        else:
            covered_x_plot_real_zero.append(x_test_scaled[i][2])
            covered_y_plot_real_zero.append(x_test_scaled[i][4])

        if y_pred[i] == y_test[i]:
            covered_x_plot_correct.append(x_test_scaled[i][2])
            covered_y_plot_correct.append(x_test_scaled[i][4])
        else:
            covered_x_plot_incorrect.append(x_test_scaled[i][2])
            covered_y_plot_incorrect.append(x_test_scaled[i][4])

    else:
        uncovered_y_pred.append(y_pred[i])
        uncovered_y_test.append(y_test[i])
        uncovered_x_test.append(x_test_scaled[i])
        print("NeighbourCount= " + str(neighbourCount))
        print("Prediction VS  Observation: " + str(y_pred[i]) + "|" + str(y_test[i]))
        print("Features: " + str(x_test_scaled[i][2]) + "|" + str(x_test_scaled[i][4]))
        print("------------------------------------------------------------------------")

        if y_pred[i] == 1:
            uncovered_x_plot_predicted_one.append(x_test_scaled[i][2])
            uncovered_y_plot_predicted_one.append(x_test_scaled[i][4])
        else:
            uncovered_x_plot_predicted_zero.append(x_test_scaled[i][2])
            uncovered_y_plot_predicted_zero.append(x_test_scaled[i][4])
        if y_test[i] == 1:
            uncovered_x_plot_real_one.append(x_test_scaled[i][2])
            uncovered_y_plot_real_one.append(x_test_scaled[i][4])
        else:
            uncovered_x_plot_real_zero.append(x_test_scaled[i][2])
            uncovered_y_plot_real_zero.append(x_test_scaled[i][4])

        if y_pred[i] == y_test[i]:
            uncovered_x_plot_correct.append(x_test_scaled[i][2])
            uncovered_y_plot_correct.append(x_test_scaled[i][4])
        else:
            uncovered_x_plot_incorrect.append(x_test_scaled[i][2])
            uncovered_y_plot_incorrect.append(x_test_scaled[i][4])

fig = plt.figure()
fig.suptitle('accuracy-location correlation', fontsize=14, fontweight='bold')
plt.plot(uncovered_x_plot_incorrect, uncovered_y_plot_incorrect, 'ro', markersize=2, label="uncovered:incorrect")
plt.plot(uncovered_x_plot_correct, uncovered_y_plot_correct, 'bo', markersize=2, label="uncovered:correct")
plt.plot(covered_x_plot_incorrect, covered_y_plot_incorrect, 'kx', markersize=2, label="covered:incorrect")
plt.plot(covered_x_plot_correct, covered_y_plot_correct, 'gx', markersize=2, label="covered:correct")

plt.legend(loc='best', numpoints=1, ncol=2, fontsize=8)
plt.axis([0, 1, 0, 1])
plt.show()

fig = plt.figure()
fig.suptitle('model prediction-location correlation', fontsize=14, fontweight='bold')
plt.plot(uncovered_x_plot_predicted_one, uncovered_y_plot_predicted_one, 'bo', markersize=2, label="uncovered:1")
plt.plot(uncovered_x_plot_predicted_zero, uncovered_y_plot_predicted_zero, 'ro', markersize=2, label="uncovered:0")
plt.plot(covered_x_plot_predicted_one, covered_y_plot_predicted_one, 'gx', markersize=2, label="covered:1")
plt.plot(covered_x_plot_predicted_zero, covered_y_plot_predicted_zero, 'kx', markersize=2, label="covered:0")

plt.legend(loc='best', numpoints=1, ncol=2, fontsize=8)
plt.axis([0, 1, 0, 1])
plt.show()

fig = plt.figure()
fig.suptitle('observation-location correlation', fontsize=14, fontweight='bold')
plt.plot(uncovered_x_plot_real_one, uncovered_y_plot_real_one, 'bo', markersize=2, label="uncovered:1")
plt.plot(uncovered_x_plot_real_zero, uncovered_y_plot_real_zero, 'ro', markersize=2, label="uncovered:0")
plt.plot(covered_x_plot_real_one, covered_y_plot_real_one, 'gx', markersize=2, label="covered:1")
plt.plot(covered_x_plot_real_zero, covered_y_plot_real_zero, 'kx', markersize=2, label="covered:0")

plt.legend(loc='best', numpoints=1, ncol=2, fontsize=8)
plt.axis([0, 1, 0, 1])
plt.show()

covered_points = pd.DataFrame(list(zip(covered_y_pred, covered_y_test)), columns=['prediction', 'observation'])
uncovered_points = pd.DataFrame(list(zip(uncovered_y_pred, uncovered_y_test)), columns=['prediction', 'observation'])

print("Covered: " + str(len(covered_points)))
print("Uncovered: " + str(len(uncovered_points)))
print("Total: " + str(len(y_test)))


def evaluate(df, status):
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
    print(str(status))
    print("Number of test items: " + str(num_values))
    print("TP: " + str(TP))
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str((TP + TN) / (TP + TN + FP + FN)))
    print("Classification Error: " + str((FP + FN) / (TP + TN + FP + FN)))
    print("Positive Predictive Value: " + str(TP / (TP + FP)))
    print("Demographic Parity: " + str((TP + FP) / (TP + TN + FP + FN)))
    print("False Positive Rate: " + str(FP / (TN + FP)))


evaluate(covered_points, "******Covered******")
evaluate(uncovered_points, "******Uncovered******")

covered_x_test = np.array(covered_x_test)
covered_y_test = np.array(covered_y_test)
uncovered_x_test = np.array(uncovered_x_test)
uncovered_y_test = np.array(uncovered_y_test)

mse, bias, var = bias_variance_decomp(clf, x_train_scaled, y_train, covered_x_test, covered_y_test, loss='mse',
                                      num_rounds=200,
                                      random_seed=1)
print("******Covered******")
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

mse, bias, var = bias_variance_decomp(clf, x_train_scaled, y_train, uncovered_x_test, uncovered_y_test, loss='mse',
                                      num_rounds=200,
                                      random_seed=1)
print("******Uncovered******")
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
