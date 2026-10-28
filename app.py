import gradio as gr
from ultralytics import YOLO

model = YOLO("models/yolo11n.onnx", task="detect")


def detect(image, conf):
    result = model.predict(image, imgsz=416, conf=conf, verbose=False)[0]
    return result.plot()[..., ::-1]


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Fotoğraf / Image"),
        gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Güven eşiği / Confidence"),
    ],
    outputs=gr.Image(label="Sonuç / Result"),
    title="Böcek ve Omurgasız Tespiti / Insect & Invertebrate Detection",
    description="YOLO11n, 12 sınıf: karınca, arı, böcek, tırtıl, solucan, kulağakaçan, çekirge, güve, sümüklüböcek, salyangoz, eşek arısı, kınkanatlı.",
)

if __name__ == "__main__":
    demo.launch()
