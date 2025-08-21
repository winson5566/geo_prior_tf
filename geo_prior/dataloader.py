# Copyright 2021 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math

import pandas as pd
import tensorflow as tf

import utils

AUTOTUNE = tf.data.AUTOTUNE

class JsonInatInputProcessor:
  def __init__(self,
              dataset_json,
              location_info_json=None,
              batch_size=1,
              loc_encode='encode_cos_sin',
              date_encode='encode_cos_sin',
              use_date_feats=True,
              use_photographers=False,
              is_training=False,
              use_data_augmentation=False,
              remove_invalid=True,
              provide_validity_info_output=False,
              max_instances_per_class=-1,
              default_empty_label=0,
              num_classes=None,
              provide_instance_id=False,
              batch_drop_remainder=True):
    self.dataset_json = dataset_json
    self.location_info_json = location_info_json
    self.batch_size = batch_size
    self.loc_encode = loc_encode
    self.date_encode = date_encode
    self.use_date_feats = use_date_feats
    self.use_photographers = use_photographers
    self.is_training = is_training
    self.use_data_augmentation = use_data_augmentation
    self.default_empty_label = default_empty_label
    self.remove_invalid = remove_invalid
    self.provide_validity_info_output = provide_validity_info_output
    self.max_instances_per_class = max_instances_per_class
    self.provide_instance_id = provide_instance_id
    self.batch_drop_remainder = batch_drop_remainder
    self.num_instances = 0
    self.num_classes = num_classes
    self.num_users = 1
    self.num_feats = 0

  def _validate_location_info_from_metadata(self, metadata_df):
    metadata = metadata_df.copy()
    if ('longitude' not in metadata.columns):
      raise RuntimeError('Logintude info does not exists on dataset_json.'
                         ' Please add to json or specify location_info_json.')
    if ('latitude' not in metadata.columns):
      raise RuntimeError('Latitude info does not exists on dataset_json.'
                         ' Please add to json or specify location_info_json.')
    if ('date' not in metadata.columns):
      raise RuntimeError('Date info does not exists on dataset_json.'
                         ' Please add to json or specify location_info_json.')

    if ('user_id' not in metadata.columns):
      metadata['user_id'] = 0
    
    if ('valid' not in metadata.columns):
      metadata['valid'] = ~metadata.longitude.isna()
    else:
      metadata['valid'] = metadata['valid'].astype('bool')
    metadata['lat'] = metadata['latitude']
    metadata['lon'] = metadata['longitude']
    metadata['date_c'] = metadata.apply(
                            lambda row: utils.date2float(row['date']), axis=1)

    return metadata

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])
    if 'annotations' in json_data.keys():
      annotations = pd.DataFrame(json_data['annotations'])
      images = pd.merge(images,
                        annotations[["image_id", "category_id"]],
                        how='left',
                        left_on='id',
                        right_on='image_id')
    else:
      images['category_id'] = self.default_empty_label
    
    num_classes = len(json_data['categories'])

    if self.location_info_json is None:
      images = self._validate_location_info_from_metadata(images)
    else:
      with tf.io.gfile.GFile(self.location_info_json, 'r') as json_file:
        json_data = json.load(json_file)
      location_info = pd.DataFrame(json_data)
      images = pd.merge(images,
                        location_info,
                        how='left',
                        on='id')

    return images, num_classes

  def _get_balanced_dataset(self, metadata):
    categories_ds = []
    categories_weights = []

    for category in list(metadata.category_id.unique()):
      cat_metadata = metadata[metadata.category_id == category]
      num_instances_cat = len(cat_metadata)

      cat_ds = tf.data.Dataset.from_tensor_slices((
                      cat_metadata.id,
                      cat_metadata.valid,
                      cat_metadata.lat,
                      cat_metadata.lon,
                      cat_metadata.date_c,
                      cat_metadata.user_id,
                      cat_metadata.category_id))
      cat_ds = cat_ds.shuffle(num_instances_cat).repeat()
      cat_weight = float(min(num_instances_cat, self.max_instances_per_class))
      categories_weights.append(cat_weight)      
      categories_ds.append(cat_ds)

    dataset = tf.data.Dataset.sample_from_datasets(categories_ds,
                                                   weights=categories_weights)
    self.num_instances = int(sum(categories_weights))

    return dataset

  def _calculate_num_features(self):
    num_feats = 0

    if self.loc_encode == 'encode_cos_sin':
      num_feats += 4
    
    if self.use_date_feats:
      if self.date_encode == 'encode_cos_sin':
        num_feats += 2
    
    self.num_feats = num_feats

  def make_source_dataset(self):
    metadata, num_classes = self._load_metadata()
    self._calculate_num_features()
    if self.num_classes is None:
      self.num_classes = num_classes

    if self.remove_invalid:
      metadata = metadata[metadata.valid].copy()
    self.num_instances = len(metadata)
    self.num_users = len(metadata.user_id.unique())
    if self.use_photographers and self.num_users < 2:
      raise RuntimeError('To add photographers branch to the model, data must'
                         ' have more than one photographer')

    if self.max_instances_per_class == -1:
      dataset = tf.data.Dataset.from_tensor_slices((
        metadata.id,
        metadata.valid,
        metadata.lat,
        metadata.lon,
        metadata.date_c,
        metadata.user_id,
        metadata.category_id))

      if self.is_training:
        dataset.shuffle(self.num_instances)
    else:
      dataset = self._get_balanced_dataset(metadata)

    def _encode_feat(feat, encode):
      if encode == 'encode_cos_sin':
        return tf.sin(math.pi*feat), tf.cos(math.pi*feat)
      else:
        raise RuntimeError('%s not implemented' % encode)

      return feat 

    def _preprocess_data(id, valid, lat, lon, date_c, user_id, category_id):
      if self.is_training and self.use_data_augmentation:
        lon, lat = utils.random_loc(lon, lat)

      lat = tf.cond(valid, lambda: lat/90.0, lambda: tf.cast(0.0, tf.float64))
      lon = tf.cond(valid, lambda: lon/180.0, lambda: tf.cast(0.0, tf.float64))
      lat = _encode_feat(lat, self.loc_encode)
      lon = _encode_feat(lon, self.loc_encode)

      if self.use_date_feats:
        date_c = tf.cond(valid, lambda: date_c, lambda: tf.cast(0.5, tf.float64))
        if self.is_training and self.use_data_augmentation:
          date_c = utils.random_date(date_c)

        date_c = date_c*2.0 - 1.0
        date_c = _encode_feat(date_c, self.date_encode)
        inputs = tf.concat([lon, lat, date_c], axis=0)
      else:
        inputs = tf.concat([lon, lat], axis=0)
      inputs = tf.cast(inputs, tf.float32)

      category_id = tf.one_hot(category_id, self.num_classes)

      if self.use_photographers:
        if self.num_users > 1:
          user_id = tf.one_hot(user_id, self.num_users)
        if self.provide_validity_info_output:
          valid = tf.cast(valid, tf.float32)
          if self.provide_instance_id:
            outputs = category_id, user_id, valid, id
          else:
            outputs = category_id, user_id, valid
        else:
          if self.provide_instance_id:
            outputs = category_id, user_id, id
          else:
            outputs = category_id, user_id
      else:
        if self.provide_validity_info_output:
          valid = tf.cast(valid, tf.float32)
          if self.provide_instance_id:
            outputs = category_id, valid, id
          else:
            outputs = category_id, valid
        else:
          if self.provide_instance_id:
            outputs = category_id, id
          else:
            outputs = category_id

      return inputs, outputs

    dataset = dataset.map(_preprocess_data, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.batch_drop_remainder)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return (dataset, self.num_instances, self.num_classes, self.num_users, \
            self.num_feats)

class RandSpatioTemporalGenerator:
  def __init__(self,
               rand_type='spherical',
               loc_encode='encode_cos_sin',
               date_encode='encode_cos_sin',
               use_date_feats=True):
    self.rand_type = rand_type
    self.loc_encode = loc_encode
    self.date_encode = date_encode
    self.use_date_feats = use_date_feats

  def _encode_feat(self, feat, encode):
    if encode == 'encode_cos_sin':
      feats = tf.concat([
        tf.sin(math.pi*feat),
        tf.cos(math.pi*feat)], axis=1)
    else:
      raise RuntimeError('%s not implemented' % encode)

    return feats

  def get_rand_samples(self, batch_size):
    if self.rand_type == 'spherical':
      rand_feats = tf.random.uniform(shape=(batch_size, 3),
                                    dtype=tf.float32)
      theta1 = 2.0*math.pi*rand_feats[:,0]
      theta2 = tf.acos(2.0*rand_feats[:,1] - 1.0)
      lat = 1.0 - 2.0*theta2/math.pi
      lon = (theta1/math.pi) - 1.0
      time = rand_feats[:,2]*2.0 - 1.0

      lon = tf.expand_dims(lon, axis=-1)
      lat = tf.expand_dims(lat, axis=-1)
      time = tf.expand_dims(time, axis=-1)
    else:
      raise RuntimeError('%s rand type not implemented' % self.rand_type)

    lon = self._encode_feat(lon, self.loc_encode)
    lat = self._encode_feat(lat, self.loc_encode)
    time = self._encode_feat(time, self.date_encode)

    if self.use_date_feats:
      return tf.concat([lon, lat, time], axis=1)
    else:
      return tf.concat([lon, lat], axis=1)
