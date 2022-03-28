import pandas as pd
import matplotlib.pyplot as plt


GROUND_TRUTH_CSV = '/home/burrowingowl/asm-nas/demo_data/pic_label.csv'
PREDICTION_TRUTH_CSV = '/home/burrowingowl/ASM_Classification/data/prediction_results_torch.csv'

# Read CSV
groundTruthData = pd.read_csv(GROUND_TRUTH_CSV)
predictionData = pd.read_csv(PREDICTION_TRUTH_CSV)

# Split CSV so Frame Number gets in it's own column
groundTruthData[['splitText', 'frameNumberJPG']
                ] = groundTruthData.pic_name.str.split("frame_", expand=True)
groundTruthData[['frameNumber', 'JPG']
                ] = groundTruthData.frameNumberJPG.str.split(".jpg", expand=True)
predictionData[['splitText', 'frameNumberJPG']
               ] = predictionData.pic_name.str.split("frame_", expand=True)
predictionData[['frameNumber', 'JPG']
               ] = predictionData.frameNumberJPG.str.split(".jpg", expand=True)

# Convert to number type and Sort by Frame Number
groundTruthData["frameNumber"] = pd.to_numeric(groundTruthData["frameNumber"])
predictionData["frameNumber"] = pd.to_numeric(predictionData["frameNumber"])
groundTruthData = groundTruthData.sort_values(by=['frameNumber'])
predictionData = predictionData.sort_values(by=['frameNumber'])

# 3 means out of frame so label should be 0
# 2 means in frame so label should be 1
groundTruthData.loc[groundTruthData["label"] == 3, 'label'] = 0
groundTruthData.loc[groundTruthData["label"] == 2, 'label'] = 1

# Plot the graph
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(groundTruthData['frameNumber'], groundTruthData['label'],
            c='b', marker="x", label='Ground Truth', alpha=0.9)
ax1.scatter(predictionData['frameNumber'], predictionData['label'],
            c='r', marker="x", label='Prediction', alpha=0.4)
plt.yticks([0, 1], ['Out-frame', 'In-frame'])
plt.xlabel("Frame Number")
plt.title("Comparison for " + groundTruthData['splitText'][0])
plt.legend()
# plt.show()
plt.savefig('/home/burrowingowl/ASM_Classification/analysis/bar.png')