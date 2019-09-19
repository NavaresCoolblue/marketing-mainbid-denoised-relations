import pandas as pd
import numpy as np

raw_data = pd.read_csv("dataset.csv", index_col=0)
fb_levels = pd.read_csv("cr_fallback_levels.csv", index_col=0)


def logit(p):
    return np.log(p) - np.log(1 - p)


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


# 1. GET DATA CORRESPONDING TO FALLBACK LEVEL
i = 0
print(fb_levels.iloc[i, :])
data = raw_data[raw_data["producttype"] == 2237].copy()

# 2. dummify dow mo
data.columns
data = pd.get_dummies(data, columns=['month', 'day_of_week'])

group_feed = [fb_levels.iloc[i, :].values[j] for j in [2, 4, 3, 5, 1, 0]]

# 3. clean null values
selected = data[pd.notnull(data['conversionrate'])]

# 4. APPLY CORRESPONDENT FILTERS
selected.shape[0] >= 1

# 5. Obtain data
selected_combined_impression_rate = list(selected['combined_impression_rate'])
selected_conversionrate = list(selected['conversionrate'])

# 5.1 Cap / floor 0s and 1s to epsilon due to logit transformation
EPSILON = 0.001  # fake value
selected.loc[selected.conversionrate == 0, 'conversionrate'] = EPSILON
selected.loc[selected.conversionrate == 1, 'conversionrate'] = 1 - EPSILON

# 5.2 CREATE X WITHOUT INTERCEPT TERM
X_wo = np.array(selected.filter(regex=r'^combined\_impression\_rate', axis=1))

# 5.3 CREATE X WITH INTERCEPT TERM
X = np.c_[np.ones(selected.shape[0]), X_wo]

# CREATE X WITHOUT INTERCEPT TERM
additional_X.shape
additional_X = np.ones(shape=(np.int(np.floor(X.shape[0] / 2)), X.shape[1]))
additional_X[:, 1] = 1.1667
X = np.concatenate((X, additional_X))

# 5.4 CREATE Y
y = np.array(selected['conversionrate'])
additional_y = np.zeros(shape=(np.int(np.floor(y.shape[0] / 2))))
additional_y[:] = EPSILON

# 5.5 transform
y = logit(np.concatenate((y, additional_y)))

# 6. Fit
coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

if coeffs[1] < 0:
    CIR_N_POINTS = 20
    conversionrate = list(
        inv_logit(coeffs[0] + coeffs[1] * np.linspace(0.05, 1.0, CIR_N_POINTS))
    )
    if max(conversionrate) / min(conversionrate) >= 2:
        conversionrate = 11111  # add default curve
else:
    conversionrate = 1111  # default curve


# 7. IF first not passed
#  ((fb_levels.iloc[i,:]['aggregation'] != 'producttype') | ((fb_levels.iloc[i,:]['aggregation'] == 'producttype') & (len(product_type_conversionrate) == 0))) & (selected.shape[0] >= ms.MIN_OBSERVATIONS_CR):
# in the original code default curve
selected_combined_impression_rate = list(selected['combined_impression_rate'])
selected_conversionrate = list(selected['conversionrate'])
conversionrate = 111  # default curve


CIR_N_POINTS + 20
combined_impression_rate = [
    round(x, 2) for x in list(np.linspace(0.05, 1.0, CIR_N_POINTS))
]

# 8. Output
feed_dict, feed_dict_input_data = utils.format_output(
    group=group_feed,
    metric='cr',
    metric_data=conversionrate,
    combined_impression_rate=combined_impression_rate,
    selected_metric_data=selected['conversionrate'],
    selected_combined_impression_rate=selected_combined_impression_rate,
)
