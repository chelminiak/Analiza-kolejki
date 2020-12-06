import os
from kolejka import Kolejka
from osoba import Osoba

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import magic
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


# flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
# flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
# flags.DEFINE_string('output', None, 'path to output video')
# flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.50, 'score threshold')


# flags.DEFINE_boolean('dont_show', True, 'dont show video output')
# flags.DEFINE_string('logs', './outputs/logs.txt', 'path to output logs')


# object_tracker.py --video ścieżka_do_video --output output_video --logs /etc/home/

# chyba jednak tak trzeba bedzie
# findPerson(video_path, output, logs)

class find:

    frame_num = 0

    def findPerson(video_path, output_video_path, log_path):

        # return 1 - wrong file format
        # return 0 - success

        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        # session = InteractiveSession(config=config)
        # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

        # resize images to
        input_size = 416
        # video_path = FLAGS.video

        # define color of boxes for categories
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 0-red, 1-green, 2-blue

        # define kolejka
        kolejka = Kolejka(0)

        # load saved model
        saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416', tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # open file to write logs
        logs = open(log_path, "w")

        # check if file is a video and begin video capture
        mime = magic.Magic(mime=True)
        filename = mime.from_file(video_path)
        if filename.find('video') != -1:
            try:
                vid = cv2.VideoCapture(video_path)
            except:
                return 1
        else:
            return 1
            # print('It is not a video, please choose another file')
            # exit()

        # alternative version?
        '''
        try:
            vid = cv2.VideoCapture(video_path)
        except:
            print("Try again with another file")
            exit()
        '''

        # out = None

        # get video ready to save locally
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

        # frame_num = 0
        # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # image = Image.fromarray(frame)
            else:
                cv2.destroyAllWindows()
                logs.close()
                return 0
                # print('Video has ended or failed, try a different video format!')
                # break
            find.frame_num += 1
            print('Frame #: ', find.frame_num)
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()
            time_in_video = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000  # in seconds

            # run detections
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.5
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # loop through objects and use class index to get class name, allow only person class
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name != 'person':
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)

            # delete detections that are not people
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(bboxes, scores, names, features)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    person_to_delete = kolejka.getOsoba(track.track_id)
                    if person_to_delete is not None:
                        person_to_delete.setKoniec(time_in_video)
                        kolejka.usunOsobe(person_to_delete)
                    continue
                bbox = track.to_tlbr()

                # check if it is new person
                if kolejka.getOsoba(track.track_id) is None:
                    # create person, detect color and add to kolejka (listaOsob)

                    # read rgb values of pixels in bbox (bbox is trimmed to get more acurate detection of color)
                    r = []
                    g = []
                    b = []
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    for x in range(int(bbox[0] + (0.1 * bbox_width)), int(bbox[2] - (0.1 * bbox_width))):
                        for y in range(int(bbox[1] + (0.1 * bbox_height)), int(bbox[3] - (0.1 * bbox_height))):
                            color = frame[y, x]
                            r.append(color[0])
                            g.append(color[1])
                            b.append(color[2])

                    r_mean = np.mean(r)
                    b_mean = np.mean(b)
                    g_mean = np.mean(g)

                    # create person for specific category depending on r, g, b values
                    # add person to kolejka
                    if b_mean == max(b_mean, g_mean, r_mean):
                        person = Osoba(track.track_id, 2, time_in_video, time_in_video,
                                       0.5 * (int(bbox[0]) + int(bbox[2])),
                                       0.5 * (int(bbox[1]) + int(bbox[3])))
                        kolejka.dodajOsobe(person)
                    elif g_mean == max(b_mean, g_mean, r_mean):
                        person = Osoba(track.track_id, 1, time_in_video, time_in_video,
                                       0.5 * ((int(bbox[0])) + int(bbox[2])), 0.5 * (int(bbox[1]) + int(bbox[3])))
                        kolejka.dodajOsobe(person)
                    else:
                        person = Osoba(track.track_id, 0, time_in_video, time_in_video,
                                       0.5 * ((int(bbox[0])) + int(bbox[2])), 0.5 * (int(bbox[1]) + int(bbox[3])))
                        kolejka.dodajOsobe(person)

                # get category of person and draw box in color of the category
                person = kolejka.getOsoba(track.track_id)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              colors[person.getKategoria()], 2)

            # show current number of people on video
            cv2.putText(frame, "Current People Count: " + str(kolejka.getLiczbaOsob()), (0, 35),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.5, (255, 255, 0), 2)

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            # result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # write to log file --> time;number of people detected;people in cat.1;people in cat.2;people in cat.3
            people_in_categories = kolejka.getLiczbaOsobKategorie()
            logs.write(
                str(time_in_video) + ";" + str(kolejka.getLiczbaOsob()) + ";" + str(
                    people_in_categories[0]) + ";" + str(
                    people_in_categories[1]) + ";" + str(people_in_categories[2]) + "\n")

            # if not FLAGS.dont_show:
            #    cv2.imshow("Output Video", result)

            # if output flag is set, save video file
            # if FLAGS.output:
            out.write(result)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

        # cv2.destroyAllWindows()
        # logs.close()


'''
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
'''
