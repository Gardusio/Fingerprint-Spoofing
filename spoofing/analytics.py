def geat_feature_mean(ds, feature_idx):
    return ds.mean(axis=1).reshape(ds.shape[0], 1)[feature_idx]

def geat_feature_var(ds, feature_idx):
    return ds.var(axis=1).reshape(ds.shape[0], 1)[feature_idx]