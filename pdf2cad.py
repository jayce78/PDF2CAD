import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import ezdxf
import tkinter as tk
from tkinter import filedialog

def process_pdf(pdf_path, status_label):
    status_label.config(text="Loading PDF...")
    root.update_idletasks()

    # Load PDF and convert first page to image
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()
    if pix.n < 4:
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    else:
        # Convert to RGB if image has alpha channel
        pix = fitz.Pixmap(fitz.csRGB, pix)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    status_label.config(text="Processing image...")
    root.update_idletasks()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    status_label.config(text="Detecting shapes...")
    root.update_idletasks()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    shapes = []
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            shapes.append(('rectangle', approx))
        elif len(approx) > 4:
            shapes.append(('circle', approx))

    status_label.config(text="Extracting text...")
    root.update_idletasks()
    d = pytesseract.image_to_data(gray, config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)

    status_label.config(text="Saving DXF file...")
    root.update_idletasks()
    doc = ezdxf.new(dxfversion='R2010')
    modelspace = doc.modelspace()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            modelspace.add_line((x1, y1), (x2, y2), dxfattribs={'layer': 'Geometry'})

    for shape, approx in shapes:
        if shape == 'rectangle':
            x, y, w, h = cv2.boundingRect(approx)
            modelspace.add_lwpolyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)], dxfattribs={'layer': 'Geometry'})
        elif shape == 'circle':
            (x, y), radius = cv2.minEnclosingCircle(approx)
            modelspace.add_circle((int(x), int(y)), int(radius), dxfattribs={'layer': 'Geometry'})

    for i in range(len(d['level'])):
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        text = d['text'][i]
        if text.strip():
            modelspace.add_text(text, dxfattribs={'height': 10, 'layer': 'Text'}).set_pos((x, y), align='LEFT')

    doc.saveas('output.dxf')
    status_label.config(text="DXF file saved as output.dxf")

def select_pdf(status_label):
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if pdf_path:
        process_pdf(pdf_path, status_label)

# GUI setup
root = tk.Tk()
root.title("PDF to DXF Converter")
root.geometry("400x400")

status_label = tk.Label(root, text="Select a PDF file to convert")
status_label.pack(pady=10)

tk.Button(root, text="Select PDF File", command=lambda: select_pdf(status_label)).pack(pady=20)

root.mainloop()
