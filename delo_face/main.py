import boto3
from PIL import Image, ImageDraw, ImageFont
import click
import csv
import os
from halo import Halo

FONT_SIZE = 24
rek = boto3.client("rekognition", region_name='ap-northeast-1')

EMOTIONS = [
    'HAPPY',
    'SAD',
    'ANGRY',
    'CONFUSED',
    'DISGUSTED',
    'SURPRISED',
    'CALM',
    'UNKNOWN',
    'FEAR'
]

@click.command(help='顔写真スナップを解析して感情分析結果をCSV出力します')
@click.option('-i', '--input', 'input_image', type=str, help='解析する顔が含まれた画像を指定します', required=True)
@click.option('-o', '--output', 'output_csv', type=str, help='解析した結果を出力するCSVファイル名を指定します', required=False, default="result.csv")
@click.option('-r', '--result', 'result_img', type=str, help='解析した結果を出力する画像ファイル名を指定します', required=False, default="result.jpg")
def main(input_image, output_csv, result_img):
    print("INFO: 顔画像解析を開始します")
    try:
        with open(input_image, "rb") as f:
            file_name = os.path.basename(input_image)
            no_ext_name = file_name.split(".")[0]
            csv_filename = result_img
            result_filename = output_csv
            if output_csv == 'result.csv':
                csv_filename = no_ext_name + "_" + output_csv
            if result_img == 'result.jpg':
                result_filename = no_ext_name + "_" + result_img
 
            with open(csv_filename, "w", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['NO', 'IS_SMILE', 'SMILE_CONFIDENTIAL'] + EMOTIONS)
                writer.writeheader()
                spinner = Halo(text='AWS Rekognitionで顔画像を解析しています...', spinner='dots')
                spinner.start()
                faces = rek.detect_faces(Image={'Bytes':f.read()}, Attributes=['ALL'])
                spinner.stop()

                im = Image.open(input_image)
                font = ImageFont.truetype("Arial.ttf", FONT_SIZE)
                width, height = im.size
                for index, face in enumerate(faces["FaceDetails"]):
                    csv_row_dict = {}
                    csv_row_dict['NO'] = index+1
                    csv_row_dict['IS_SMILE'] = face["Smile"]["Value"]
                    csv_row_dict['SMILE_CONFIDENTIAL'] = face["Smile"]["Confidence"]
                    for emotion in face["Emotions"]:
                        csv_row_dict[emotion['Type']] = emotion['Confidence']
                    dr = ImageDraw.Draw(im)
                    left = width * face["BoundingBox"]["Left"]
                    top = height * face["BoundingBox"]["Top"]
                    fw = width * face["BoundingBox"]["Width"]
                    fh = height * face["BoundingBox"]["Height"]
                    dr.rectangle((left, top, left+fw, top+fh ), outline=(255, 0, 0))
                    dr.text((left, top), "No.{}".format(index+1), fill=(255, 0, 0), font=font)
                    writer.writerow(csv_row_dict)
                im.save(result_filename, quolity=100)

    except FileNotFoundError as err:
        print("ERROR: ファイル {} が見つかりません".format(input_image))

    print("INFO: 顔画像解析を終了します")
# img = cv2.imread("party.jpg")

# ret, buf = cv2.imencode('.jpg', img)

# Amazon RekognitionにAPIを投げる
# faces = rek.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])

if __name__ == '__main__':
    main()