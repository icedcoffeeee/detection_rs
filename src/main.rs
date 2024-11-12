use core::str;
use std::sync::mpsc::TryRecvError;
use std::{fs, sync::mpsc, thread};

use opencv::{core::*, imgproc, prelude::*};
use opencv::{dnn, highgui, videoio};

const WEIGHTS_FILE: &str = "data/yolov3-tiny.weights";
const CONFIG_FILE: &str = "data/yolov3-tiny.cfg";
const CLASSES_FILE: &str = "data/yolov3.txt";

type Detections = (Vector<i32>, Vector<f32>, Vector<Rect_<i32>>);

fn main() {
    let (t_feed, r_feed) = mpsc::channel();
    let (t_detections, r_detections) = mpsc::channel();

    let classes = fs::read(CLASSES_FILE).unwrap();
    let classes: Vec<&str> = str::from_utf8(&classes).unwrap().split("\n").collect();

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_V4L).unwrap();
    let mut feed = Mat::default();

    thread::spawn(move || {
        let mut model = dnn::DetectionModel::new(WEIGHTS_FILE, CONFIG_FILE).unwrap();
        let mut init = false;
        loop {
            let feed: Mat = match r_feed.try_recv() {
                Ok(feed) => feed,
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => break,
            };
            if !init {
                init = true;
                model.set_input_size(feed.size().unwrap()).unwrap();
                model.set_input_scale(0.001).unwrap();
                model.set_input_mean([0.; 4].into()).unwrap();
                model.set_input_swap_rb(true).unwrap();
            }
            let mut detections = Detections::default();
            let (ref mut class_ids, ref mut scores, ref mut rects) = &mut detections;
            model.detect_def(&feed, class_ids, scores, rects).unwrap();
            let _ = t_detections.send(detections);
        }
    });

    let mut detections: Option<Detections> = None;
    let mut first = true;
    // while not escape key
    while highgui::wait_key(1).unwrap() != 27 {
        cam.read(&mut feed).unwrap();
        if first {
            first = false;
            t_feed.send(feed.clone()).unwrap();
        }

        if let Ok(det) = r_detections.try_recv() {
            detections = Some(det);
            t_feed.send(feed.clone()).unwrap();
        }

        if let Some(ref detections) = detections {
            draw_bounding_boxes(&mut feed, &detections, &classes);
        }

        highgui::imshow("feed", &feed).unwrap();
    }
}

fn draw_bounding_boxes(mat: &mut Mat, detections: &Detections, classes: &Vec<&str>) {
    let (class_ids, scores, rects) = detections;
    let mut indices = Vector::<i32>::new();
    dnn::nms_boxes_def(&rects, &scores, 0.5, 0.1, &mut indices).unwrap();

    for i in indices {
        let rect: Rect = rects.to_vec()[i as usize];
        let label = classes[class_ids.to_vec()[i as usize] as usize];
        let color: VecN<f64, 4> = [0., 0., 255., 255.].into();
        imgproc::rectangle_def(mat, rect, color).unwrap();

        let text_org = Point::new(rect.x - 10, rect.y - 10);
        let font = imgproc::FONT_HERSHEY_SIMPLEX;
        imgproc::put_text_def(mat, label, text_org, font, 0.5, color).unwrap();
    }
}
