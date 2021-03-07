import pandas as pd
import csv
import warnings
import warnings
warnings.filterwarnings('ignore')

#importing custom module
from NN_H1 import *

np.random.seed(20)


#Importing final trained model (weights + parameters)
params = pd.read_csv("best_set.csv", sep=",").to_dict("r")[0]
trained_model = Network(params["Ni"], params["Nh1"], params["Nh2"], params["No"], params["act_f"])
WBh1 = np.loadtxt("WBh1.txt")
WBh2 = np.loadtxt("WBh2.txt")
WBo = np.loadtxt("WBo.txt").reshape(-1,1)
trained_model.WBh1=WBh1
trained_model.WBh2=WBh2
trained_model.WBo=WBo


#Read test_set
test_set = np.loadtxt("test_set.txt",delimiter=",")
xtest, ytest = get_XY(test_set)


#Computing test set score
y_test_est = trained_model.forward(xtest)
test_loss = (y_test_est - ytest)**2
print("Mean test loss (MSE):")
mean_test_loss = np.mean(test_loss)
print("%.4f" %(mean_test_loss))

fig, ax = plt.subplots(1,1, figsize=(10,7))
ax.plot(test_loss)
ax.set_ylabel("MSE", fontsize=14)
ax.set_xlabel("Samples", fontsize=14)
ax.axhline(mean_test_loss, linestyle="--", label="Mean Loss: %.4f"%(mean_test_loss))
ax.set_title("Test Losses", fontsize=17)
ax.legend(fontsize=14)
fig.tight_layout()
plt.show()

