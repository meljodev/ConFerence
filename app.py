

import multiprocessing
from transformers import AutoTokenizer

from config import config
from core import train, test, checkpoint_callback
from inference import inference

tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL, 
                                          num_labels=20, 
                                          problem_type="multi_label_classification")

if __name__=="__main__": 
    multiprocessing.freeze_support()
    
    model = train(tokenizer)
    threshold = test(model, tokenizer)

    threshold = 0.8
    intents = inference('''Contact: سلام من الان بیمه مسافرتی خریدم کی برام ارسال میشه چون عجله دارم, User: سلام
امیدوارم روز خوبی رو سپری کرده باشید😊در خدمتتونم, User: پس از خرید و بررسی اطلاعات بیمه شما در صف صدور قرار می گیره و و در صورت نیاز به پیگیری و یا وجود مشکلی در اطلاعات همکاران با شما تماس میگیرند😊
توجه داشته باشید در صورتی که خریدتون قبل از ساعت 21 انجام بشه و مشکلی وجود نداشته باشه بیمتون صادر میشه 😍
اگر بررسی کردید و به نتیجه ای نرسیدید من همینجا در خدمتتون هستم ، همچنین میتونین با شماره 02191020345 تماس بگیرین همکارانمون در خدمتتون هستن 😊''', tokenizer, checkpoint_callback, threshold)

