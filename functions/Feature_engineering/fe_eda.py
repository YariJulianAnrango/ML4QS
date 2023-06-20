import pandas as pd
import matplotlib.pyplot as plt


features = ['heart_rmax_l96', 'heart_rstd_l2', 'heart_rmin_l40', 'heart_rstd_l97', 'heart_rmin_l50', 'value_heart_freq_0.211_Hz_ws_166', 'steps_rmin_l5', 'heart_rstd_l37', 'heart_rsum_l68', 'value_heart_freq_0.09_Hz_ws_166', 'heart_rmin_l37', 'heart_rstd_l95', 'value_heart_freq_0.171_Hz_ws_166', 'value_heart_freq_0.422_Hz_ws_166', 'steps_rmin_l82', 'steps_rmin_l70', 'steps_rmin_l78', 'value_heart_freq_0.392_Hz_ws_166', 'steps_rmin_l19', 'steps_rmin_l76']
scores = [0.468609, 0.571214, 0.580564, 0.589664, 0.594507, 0.595926, 0.596260, 0.597763, 0.600267, 0.601019, 0.601853, 0.603022, 0.603523, 0.603607, 0.603774, 0.603774, 0.603774, 0.603774, 0.603941, 0.603774]

df = pd.DataFrame({'values':scores, 'features':features})

plt.plot(pd.Categorical(df.index), df['values'])
plt.title('Accuracy of model for feature selection')
plt.xlabel('Feature amount')
plt.ylabel('Accuracy')
plt.locator_params(axis="x", integer=True)
plt.savefig('feature_selection_acc.png')
