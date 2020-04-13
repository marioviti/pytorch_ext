import io
import numpy as np
from PIL import Image
import PIL
try:
    import tensorflow as tf
except Exception as E:
    print('warning: tensorboard unavailable, cause: no tensorflow installed.')

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
        
    def log_text(self, tag, text, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, text=text)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
        
    def log_histogram(self, tag, values, global_step, bins=10):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_image_raw(self, tag, img, global_step):
        s = io.BytesIO()
        new_p = Image.fromarray(img)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(s, format='png')
        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
        
    def log_image(self, tag, image, global_step):
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        min_,max_ = np.min(image), np.max(image)
        image = (image - min_)/(max_ - min_)
        image *= 255
        image = image.astype(np.uint8)
        self.log_image_raw(tag, image, global_step)
        
    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()


def read_values_from_event_file(event_file_path, tags):
    out = { tag : [] for tag in tags }
    for e in tf.train.summary_iterator(event_file_path):
        for v in e.summary.value:
            for tag in tags:
                if v.tag == tag:
                    out[tag] += [v.simple_value]
    return out

def read_images_from_event_file(event_file_path, tags):
    out = { tag : [] for tag in tags }
    
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        for e in tf.train.summary_iterator(event_file_path):
            for v in e.summary.value:
                for tag in tags:
                    if v.tag == tag:
                        im = im_tf.eval({image_str: v.image.encoded_image_string})
                        out[tag] += [im]
    return out