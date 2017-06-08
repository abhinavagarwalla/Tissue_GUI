from nets import unet_definition, inception_resnet_v2

networks_map = {'unet': unet_definition.UNet,
                'inception_resnet_v2': inception_resnet_v2.InceptionResnetV2,
               }

def get_network_fn(name, images, num_classes=None, is_training=False):
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  cls = networks_map[name]()
  return cls.model(images, num_classes, is_training=is_training)