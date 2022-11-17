import tensorflow_datasets as tfds
import tensorflow as tf

class FGVCDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.1.0')
    
    def _split_generators(self, dl_manager):
        arr = [
            tfds.core.SplitGenerator(name=f'train',gen_kwargs={"split":"train"}),
            tfds.core.SplitGenerator(name=f'test',gen_kwargs={"split":"test"})
        ]
        return arr
    
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=(""),
            #disable_shuffling=True,
            features=tfds.features.FeaturesDict({
                "img": tfds.features.Image(encoding_format='jpeg'),#dtype=tf.uint8,shape=(self.WIDTH,self.HEIGHT,3),
                "name": tfds.features.Tensor(dtype=tf.string,shape=()),
                "cultivar": tfds.features.Tensor(dtype=tf.string,shape=()),
                "target": tfds.features.Tensor(dtype=tf.int32,shape=()),
            }),
        )
    
    def _generate_examples(self,**args):
        print(args)
        split = args["split"]
        
        if split == 'train':
            for i in range(len(self.train_df)):
                row = self.train_df.iloc[i]
                img = row.fullpath
                yield i, {
                    'img':img,
                    'cultivar':'',
                    'name':row.image,
                    'target':row.target,
                }
                
        if split == 'test':
            for i in range(len(self.test_df)):
                row = self.test_df.iloc[i]
                img = row.fullpath
                yield i, {
                    'img':img,
                    'name':row.image,
                    'cultivar':'',
                    'target':-1,
                }
