import io
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_pdf(annotated_img, details):
    ret, img_encoded = cv2.imencode('.jpg', annotated_img)
    if not ret:
        raise Exception("Failed to encode image for PDF generation.")
    img_bytes = img_encoded.tobytes()
    image_reader = ImageReader(io.BytesIO(img_bytes))
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    margin = 50
    img_width = page_width - 2 * margin
    orig_w, orig_h = image_reader.getSize()
    aspect_ratio = orig_h / float(orig_w)
    img_height = img_width * aspect_ratio

    # Draw the annotated image at the top of the first page
    c.drawImage(
        image_reader, 
        margin, 
        page_height - margin - img_height, 
        width=img_width, 
        height=img_height
    )

    # Prepare to write text below the image
    text_start = page_height - margin - img_height - 20
    text = c.beginText(margin, text_start)
    text.setFont("Helvetica", 12)

    # Write the detection summary
    text.textLine(f"Total Detections: {len(details)}")
    text.textLine("")

    for idx, d in enumerate(details, 1):
        lines_for_detection = [
            f"Detection #{idx}: {d['label']}",
            f"    Coordinates: {d['coordinates']}",
            f"    Dimensions: {d['dimensions'][0]} x {d['dimensions'][1]}",
            ""  # Blank line between detections
        ]

        for line in lines_for_detection:
            # Check if there is enough room, else start a new page
            if text.getY() <= margin:
                c.drawText(text)  # Complete the current text block
                c.showPage()      # Start a new page
                text = c.beginText(margin, page_height - margin)
                text.setFont("Helvetica", 12)
            text.textLine(line)

    # Finalize the PDF
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
