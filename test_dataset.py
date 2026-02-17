from dataset import RAVDESSDataset

ds = RAVDESSDataset("data/RAVDESS")
print("Samples:", len(ds))

x, y = ds[0]
print("MFCC shape:", x.shape)
print("Label:", y)
