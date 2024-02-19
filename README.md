# SST-CLT
The implementation source code and demo for Spatial–spectral–temporal deep regression model with convolutional long short-term memory and transformer for the large-area mapping of mangrove canopy height by using Sentinel-1 and Sentinel-2 data (SST-CLT)
https://doi.org/10.1109/TGRS.2024.3362788

Note: This implementation is based on the best trained model. There will be differences in the results compared to those in the paper because the results reported in the paper are the average results of the MAE and RMSE (the SST-CLT model was trained thrice).

# Demo
1. This code was tested on Windows 10, with:
   -  tensorflow-gpu == 2.10.0
   -  keras == 2.10.0
   -  scikit-learn == 1.2.2
   -  numpy == 1.26.0
   -  matplotlib == 3.8.0
3. Please download the best trained model and placed it into "./Model" folder : https://drive.google.com/drive/folders/1vTKxIOxJG7OaJ5lZu4vugfDzjJaU0QW0?usp=sharing 
2. Run this following code for Demo. Please select the study area: ENP or CHPSP
```bash
python demo.py --study_area "ENP"
```
# Full Dataset
This Demo is just used testing dataset. If someone want to get the Full dataset for ENP and CHPSP dataset, please fill in the following form: [Link](https://forms.gle/qhnqSqhjwzrC5u916)

# Acknowledgements
The implementation of SWINTF is benefited from: [Link](https://github.com/yingkaisha/keras-vision-transformer)

# Contact
Email: ilhamjamaluddin@g.ncu.edu.tw


