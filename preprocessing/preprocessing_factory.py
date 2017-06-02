from preprocessing import stain_normalisation

preprocessing_map = {'stain_norm': stain_normalisation.StrainNormalisation,
               }

def get_preprocessing_fn(name):
  if name not in preprocessing_map:
    raise ValueError('Name of Preprocessing unknown %s' % name)
  return preprocessing_map[name]()