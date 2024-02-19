import tensorflow as tf
from sstclt_model import sst_clt
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse

def test(args):
    study_area = args.study_area

    ## --Load Test Data-- #
    S1_test_path = "./Data/" + study_area + "_S1_test_data_20.npy"
    S2_test_path = "./Data/" + study_area + "_S2_test_data_20.npy"
    label_test_path = "./Data/" + study_area + "_test_label_20.npy"

    S1_test_data = np.load(S1_test_path)
    S2_test_data = np.load(S2_test_path)
    test_label = np.load(label_test_path)
    print('Testing data:',(S1_test_data.shape, S2_test_data.shape, test_label.shape))


    ## --SST-CLT Model Construction-- #
    model_weight = "./Model/" + study_area + "_Best_SSTCLT.h5"
    input_shape_S1 = 3, 15, 15, 2
    input_shape_S2 = 15, 15, 7
    model = sst_clt(input_shape_S1, input_shape_S2)
    model.load_weights(model_weight)
    model.summary()


    ## --MAE and RMSE Testing Result-- #
    predicted_test = model.predict([S1_test_data,S2_test_data])
    predicted_test = predicted_test.reshape(predicted_test.shape[0]*predicted_test.shape[1]*predicted_test.shape[2]*predicted_test.shape[3])
    test_label = test_label.reshape(test_label.shape[0]*test_label.shape[1]*test_label.shape[2])
    MAE = metrics.mean_absolute_error(test_label , predicted_test)
    print('Model test_data MAE = ', MAE)
    RMSE = metrics.mean_squared_error(test_label , predicted_test, squared=False)
    print('Model test_data RMSE = ', RMSE)


    ## --Scatter Plot Fig.8-- #
    fig, ax = plt.subplots()
    ax.scatter(test_label, predicted_test, s=5, facecolors='none', edgecolors='red')
    ax.plot([test_label.min(), test_label.max()], [test_label.min(), test_label.max()], 'k--', lw=2)
    ax.set_xlabel('Ground truth (m)')
    ax.set_ylabel('Predictions (m)')
    plt.savefig('scatter_plot.png', dpi=600)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='SST-CLT: For Mangrove Canopy Height Mapping')
    parser.add_argument("--study_area", type=str, default="ENP", help='Select study area: ENP or CHPSP')

    args = parser.parse_args()
    test(args)

if __name__ == "__main__":
    main()