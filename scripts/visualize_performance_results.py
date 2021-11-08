import pickle

import numpy as np
from matplotlib import pyplot as plt

from kad.kad_utils.kad_utils import customize_matplotlib_for_paper

MODEL: str = "SarimaModel"

customize_matplotlib_for_paper()

request_dur_by_length = {}
with (open("eval_res/" + MODEL + ".pkl", "rb")) as openfile:
    while True:
        try:
            request_dur_by_length = pickle.load(openfile)
        except EOFError:
            break

# AUTOENCODER
# request_dur_by_length = {61: [8.219184, 16.374639, 10.640346, 13.527358, 19.139709],
#                          121: [23.681103, 9.943325, 29.636961, 24.337622, 17.339097],
#                          241: [12.4076, 40.846229, 16.074862, 19.14179, 25.812835],
#                          481: [23.488589, 21.393581, 49.063132, 23.783926, 59.264398],
#                          961: [38.295142, 118.208349, 37.833973, 117.355759, 49.002452],
#                          1921: [50.156854, 89.612206, 76.830374, 99.493633, 194.265975]}

# ALL
request_dur_by_length = {61: [13.397192, 14.145639, 12.75082, 15.695836, 15.556507],
                         121: [15.23407, 15.453286, 16.709555, 16.710648, 17.376834],
                         241: [18.450905, 26.59773, 22.146138, 36.194055, 33.095001],
                         481: [40.484818, 40.865191, 41.576762, 50.031476],
                         961: [35.980787, 34.989967, 35.187627, 35.764971, 37.068362]}
                         # 1921: [599.419279, 203.741972, 563.862331, 261.997669, 249.165666]}

print(request_dur_by_length)

lens_train_df = []
means = []
errs = []

for key in request_dur_by_length:
    lens_train_df.append(key)
    means.append(np.mean(request_dur_by_length[key]))
    errs.append(np.std(request_dur_by_length[key]))

plt.figure(figsize=(20, 10))
plt.xlabel("Training set length")
plt.ylabel("Training phase duration [s]")
plt.errorbar(lens_train_df, means, yerr=errs, fmt="o")
plt.scatter(lens_train_df, means, s=300)
plt.grid()
plt.xlim(xmin=0.0)
plt.ylim(ymin=0.0)
plt.show()
# plt.savefig(MODEL + ".png")

print(request_dur_by_length)
