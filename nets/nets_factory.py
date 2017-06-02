from nets import unet_definition

networks_map = {'unet': unet_definition.UNet,
               }

def get_network_fn(name, images, is_training=False):
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  cls = networks_map[name]()
  return cls.model(images, is_training=is_training)
