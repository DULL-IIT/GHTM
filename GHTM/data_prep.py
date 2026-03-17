from bnlp.corpus.corpus import BengaliCorpus as bcorpus
from bnlp.cleantext.clean import CleanText
from bnlp.tokenizer.basic import BasicTokenizer
from bkit.lemmatizer import lemmatize
import pandas as pd
import re
import pickle 
import random
import os
from gensim import corpora
import argparse

def data_loading(file_path: str, text_column: str):
    """

    Supported formats:
        - .csv
        - .jsonl / .json
        - .xlsx / .xls
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)

    elif ext in [".jsonl", ".json"]:
        df = pd.read_json(file_path, lines=True)

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            "Supported formats are: .csv, .jsonl, .json, .xlsx, .xls"
        )

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    data = df[text_column].dropna().astype(str).tolist()
    random.shuffle(data)

    return data

def clean_corpus(text, stopwords):
    
    punctuations = bcorpus.punctuations
    vowels = bcorpus.vowels
    
    tokenizer = BasicTokenizer()

    clean_text = CleanText(
       fix_unicode=True,
       unicode_norm=True,
       unicode_norm_form="NFKC",
       remove_url=True,
       remove_email=True,
       remove_emoji=True,
       remove_number=True,
       remove_digits=True,
    
       replace_with_url="",
       replace_with_email="",
       replace_with_number="",
       replace_with_digit="",

    )
   
    
    cleaned = clean_text(text)    
    cleaned_text = re.sub(r'[^\u0980-\u09FF০-৯\s!?,।]', ' ', cleaned)
    cleaned_text = re.sub(r'[০-৯]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    lemmatized = lemmatize(cleaned_text)
    tokens = tokenizer(lemmatized)

    tokens = [word for word in tokens if word not in stopwords]
    tokens = [word for word in tokens if word not in punctuations]
    tokens = [word for word in tokens if word not in vowels]
    tokens = [word for word in tokens if word not in bcorpus.letters]
    tokens = [word for word in tokens if len(word) > 1]


    return tokens

def bn_stopwords():
    with open('./data/bn_stopwords.pickle', 'rb') as f:
        stopwords_3 = pickle.load(f)

    stopwords_1 = bcorpus.stopwords
    stopwords_2 = "অবশ্য অনেক অনেকে অনেকেই অন্তত অথবা অথচ অর্থাত আসা অন্য আজ আছে আপনার আপনি আবার আমরা আমাকে দেয়া আমাদের পারা রওয়া আমার আমি আরও আরো আর আগে আগেই আই অতএব আগামী অবধি অনুযায়ী আদ্যভাগে এই একই একে একটি এখন এখনও এখানে এখানেই এটি এটা এটাই এতটাই এবং একবার এবার এদের এঁদের এমন এমনকী এল এর এরা এঁরা এস এত এতে এসে একে এক এ ঐ ই ইহা ইত্যাদি উনি উপর উপরে উচিত ও ওই ওর ওরা ওঁর ওঁরা ওকে ওদের ওঁদের ওখানে কত কবে করতে কয়েক কয়েকটি করবে করলেন করার কারও করা করি করিয়ে করার করাই করলে করলেন করিতে করিয়া করেছিলেন করছে করছেন করেছেন করেছে করেন করবেন করায় করে করেই কাছ কাছে কাজে কারণ কিছু কিছুই কিন্তু কিংবা কি কী কেউ কেউই কাউকে কেন কে কোনও কোনো কোন কখনও ক্ষেত্রে খুব গুলি গিয়ে গিয়েছে গেছে গেল গেলে গোটা চলে ছাড়া ছাড়াও ছিলেন ছিল জন্য জানা ঠিক তিনি তিনঐ তিনিও তখন তবে তবু তাঁদের তাঁাহারা তাঁরা তাঁর তাঁকে তাই তেমন তাকে তাহা তাহাতে তাহার তাদের তারপর তারা তারৈ তার তাহলে তিনি তা তাও তাতে তো তত তুমি তোমার তথা থাকে থাকা থাকায় থেকে থেকেও থাকবে থাকেন থাকবেন থেকেই দিকে দিতে দিয়ে দিয়েছে দিয়েছেন দিলেন দু দুটি দুটো দেয় দেওয়া দেওয়ার দেখা দেখে দেখতে দ্বারা ধরে ধরা নয় নানা না নাকি নাগাদ নিতে নিজে নিজেই নিজের নিজেদের নিয়ে নেওয়া নেওয়ার নেই নাই পক্ষে পর্যন্ত পাওয়া পারেন পারি পারে পরে পরেই পরেও পর পেয়ে প্রতি প্রভৃতি প্রায় ফের ফলে ফিরে ব্যবহার বলতে বললেন বলেছেন বলল বলা বলেন বলে বহু বসে বার বা বিনা বরং বদলে বাদে বার বিশেষ বিভিন্ন বিষয়টি ব্যবহার ব্যাপারে ভাবে ভাবেই মধ্যে মধ্যেই মধ্যেও মধ্যভাগে মাধ্যমে মাত্র মতো মতোই মোটেই যখন যদি যদিও যাবে যায় যাকে যাওয়া যাওয়ার যত যতটা যা যার যারা যাঁর যাঁরা যাদের যান যাচ্ছে যেতে যাতে যেন যেমন যেখানে যিনি যে রেখে রাখা রয়েছে রকম শুধু সঙ্গে সঙ্গেও সমস্ত সব সবার সহ সুতরাং সহিত সেই সেটা সেটি সেটাই সেটাও সম্প্রতি সেখান সেখানে সে স্পষ্ট স্বয়ং হইতে হইবে হৈলে হইয়া হচ্ছে হত হতে হতেই হবে হবেন হয়েছিল হয়েছে হয়েছেন হয়ে হয়নি হয় হয়েই হয়তো হল হলে হলেই হলেও হলো হিসাবে হওয়া হওয়ার হওয়ায় হন হোক জন জনকে জনের জানতে জানায় জানিয়ে জানানো জানিয়েছে জন্য জন্যওজে জে বেশ দেন তুলে ছিলেন চান চায় চেয়ে মোট যথেষ্ট টি অতএব অথচ অথবা অনুযায়ী অনেক অনেকে অনেকেই অন্তত অন্য অবধি অবশ্য অর্থাত আই আগামী আগে আগেই আছে আজ আদ্যভাগে আপনার আপনি আবার আমরা আমাকে আমাদের আমার আমি আর আরও ই ইত্যাদি ইহা উচিত উত্তর উনি উপর উপরে এ এঁদের এঁরা এই একই একটি একবার একে এক্ এখন এখনও এখানে এখানেই এটা এটাই এটি এত এতটাই এতে এদের এব এবং এবার এমন এমনকী এমনি এর এরা এল এস এসে ঐ ও ওঁদের ওঁর ওঁরা ওই ওকে ওখানে ওদের ওর ওরা কখনও কত কবে কমনে কয়েক কয়েকটি করছে করছেন করতে করবে করবেন করলে করলেন করা করাই করায় করার করি করিতে করিয়া করিয়ে থাকবেন থাকা থাকায় থাকে থাকেন থেকে থেকেই থেকেও দিকে দিতে দিন দিয়ে দিয়েছে দিয়েছেন দিলেন দু দুই দুটি দুটো দেওয়া দেওয়ার দেওয়া দেখতে দেখা দেখে দেন দেয় দ্বারা ধরা ধরে ধামার নতুন নয় না নাই নাকি নাগাদ নানা নিজে নিজেই নিজেদের নিজের নিতে নিয়ে নিয়ে নেই নেওয়া নেওয়ার নেওয়া নয় পক্ষে পর পরে পরেই পরেও পর্যন্ত পাওয়া পাচ পারি পারে পারেন পি পেয়ে পেয়্র্ প্রতি প্রথম প্রভৃতি প্রযন্ত প্রাথমিক প্রায় প্রায় ফলে ফিরে ফের বক্তব্য বদলে বন বরং বলতে বলল বললেন বলা বলে বলেছেন বলেন বসে হু বা বাদে বার বি বিনা বিভিন্ন বিশেষ বিষয়টি  বেশ বেশি ব্যবহার ব্যাপারে ভাবে ভাবেই মতো মতোই মধ্যভাগে মধ্যে মধ্যেই মধ্যেও মনে মাত্র মাধ্যমে মোট মোটেই যখন যত যতটা যথেষ্ট যদি যদিও যা যাঁর যাঁরা যাওয়া যাওয়ার যাওয়া যাকে যাচ্ছে যাতে যাদের যান যাবে যায় যার যারা যিনি যে যেখানে যেতে যেন যেমন র রকম রয়েছে রাখা রেখে লক্ষ শুধু শুরু সঙ্গে সঙ্গেও সব সবার সমস্ত সম্প্রতি সহ সহিত সাধারণ সামনে সি সুতরাং সে সেই নয় না হয় হয়ে হয়েছে হয়েছিলেন  ঃ মিথ্যা শুরু"
    stopwords_4 = ['জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল','মে', 'জুন','জুলাই', 'অগাস্ট', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর', 'নভেম্বর', 'ডিসেম্বর']
    stopwords_5 = ['রবিবার', 'সোমবার', 'মঙ্গলবার', 'বুধবার', 'বৃহস্পতিবার', 'শুক্রবার', 'শনিবার']
    stopwords = list(set(stopwords_1 + stopwords_2.split() + stopwords_3 + stopwords_4 + stopwords_5))

    return stopwords

def prepare_data(data, stopwords):
    
    processed_data = [clean_corpus(doc, stopwords) for doc in data]
    print(f'Processed {len(processed_data)} text documents.')
    dictionary = corpora.Dictionary(processed_data)
    print(f'Vocabulary: {dictionary}')

    with open('./data/processed_data.pickle', 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Processed data stored in: {os.getcwd()+'/data/processed_data.pickle'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='../datasets/NCTBText/NCTBText.csv')
    parser.add_argument('--text_column', type=str, default='text')
    args = parser.parse_args()

    data = data_loading(args.file_path, args.text_column)
    stopwords = bn_stopwords()
    prepare_data(data, stopwords)