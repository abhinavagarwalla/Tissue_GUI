from preprocessing import stain_normalisation, camelyon_preprocessing

preprocessing_map = {'stain_norm': stain_normalisation.StrainNormalisation,
                     'camelyon': camelyon_preprocessing.CamelyonPreprocessing,
               }

def get_preprocessing_fn(name):
  if name not in preprocessing_map:
    raise ValueError('Name of Preprocessing unknown %s' % name)
  return preprocessing_map[name]()