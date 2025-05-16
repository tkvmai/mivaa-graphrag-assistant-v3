import random
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet
# --- Import Paragraph, Spacer, AND Frame ---
from reportlab.platypus import Paragraph, Spacer, Frame
from reportlab.lib.enums import TA_LEFT
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_TEXT_FILE = "1628-A22.txt"  # Input text file name (Make sure this file exists)
OUTPUT_PDF_FILE = "scanned_output.pdf" # Output PDF file name
FONT_NAME = "Courier" # Use a monospaced font for a 'typewritten' look
FONT_SIZE = 10
MAX_ROTATION_DEGREES = 0.5 # Max random rotation per page (adjust for more/less effect)
# --- End Configuration ---

def create_scanned_pdf(input_filename: str, output_filename: str):
    """
    Reads text from a file and creates a PDF with a simulated scanned look.

    Args:
        input_filename: Path to the input text file.
        output_filename: Path where the output PDF will be saved.
    """
    logging.info(f"Starting PDF generation from '{input_filename}' to '{output_filename}'")

    # --- Read Input Text ---
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            text_content = f.read()
        logging.info(f"Successfully read text file '{input_filename}'.")
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{input_filename}'")
        print(f"Error: Input file not found at '{input_filename}'")
        return
    except Exception as e:
        logging.error(f"Error reading input file '{input_filename}': {e}")
        print(f"Error reading input file '{input_filename}': {e}")
        return

    # --- Setup PDF ---
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter # Page dimensions

    # --- Setup Styles ---
    styles = getSampleStyleSheet()
    # Create a custom style using Courier
    style = styles['Normal']
    style.fontName = FONT_NAME
    style.fontSize = FONT_SIZE
    style.alignment = TA_LEFT
    style.leading = FONT_SIZE * 1.4 # Line spacing

    # --- Prepare Paragraphs (Flowables) ---
    # Split text into paragraphs based on double newlines to preserve structure
    text_paragraphs = text_content.split('\n\n')
    story = [] # List to hold ReportLab Flowables
    for para_text in text_paragraphs:
        if para_text.strip(): # Avoid empty paragraphs
            # Replace single newlines within a paragraph with <br/> tags for ReportLab Paragraph
            formatted_para = para_text.replace('\n', '<br/>')
            story.append(Paragraph(formatted_para, style))
            # Add a small space between original paragraphs
            story.append(Spacer(1, 0.1*inch))

    # --- Build PDF Page by Page ---
    # Define margins and frame dimensions
    margin = inch
    frame_width = width - 2 * margin
    frame_height = height - 2 * margin
    frame_x = margin
    frame_y = margin

    logging.info("Building PDF pages...")
    page_num = 1
    while story:
        logging.debug(f"Building page {page_num}...")
        # --- Apply Scan Effect: Rotation ---
        # Save the current state (includes transformations, colors, fonts etc.)
        c.saveState()
        # Translate origin to center for rotation (makes rotation around page center)
        c.translate(width / 2, height / 2)
        # Apply random slight rotation
        rotation = random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
        c.rotate(rotation)
        # Translate back
        c.translate(-width / 2, -height / 2)

        # --- Create Frame and Draw Story ---
        # --- FIXED: Instantiate Frame correctly ---
        frame = Frame(
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            leftPadding=0, # Minimal padding within the frame
            bottomPadding=0,
            rightPadding=0,
            topPadding=0,
            showBoundary=0 # Set to 1 to see the frame for debugging
        )
        # Draw the story elements (Paragraphs, Spacers) that fit on the current page's frame
        # frame.addFromList consumes items from the story list
        remaining_story = frame.addFromList(story, c)

        # --- Restore state (removes rotation and other changes for next page setup) ---
        c.restoreState()

        # --- Check if content remains ---
        if remaining_story == story and story: # Check story is not empty
            # Handle case where a single flowable (Paragraph/Spacer) is too large for the frame
            logging.warning(f"Content on page {page_num} might be too large to fit in the frame. Skipping problematic flowable: {story[0].__class__.__name__}")
            # For simplicity, skip the first remaining item to avoid an infinite loop
            story = story[1:]
            if not story: # If that was the last item, break
                 break
            # If we skipped something, force a new page to try the next item
            c.showPage()
            page_num += 1
            continue # Go to the start of the while loop for the next page

        else:
            story = remaining_story # Update story with remaining elements

        # --- Finish Page ---
        if story: # Only add a new page if there's more content
            c.showPage()
            page_num += 1
        else:
            break # No more content

    # --- Save PDF ---
    try:
        c.save()
        logging.info(f"Successfully created PDF: '{output_filename}' with {page_num} pages.")
        print(f"Successfully created PDF: '{output_filename}'")
    except Exception as e:
        logging.error(f"Error saving PDF file '{output_filename}': {e}")
        print(f"Error saving PDF file '{output_filename}': {e}")

# --- Run the function ---
if __name__ == "__main__":
    # Ensure the input file exists before running
    if not os.path.exists(INPUT_TEXT_FILE):
        print(f"Error: Input file '{INPUT_TEXT_FILE}' not found in the current directory.")
        print("Please make sure the text file is present before running the script.")
    else:
        create_scanned_pdf(INPUT_TEXT_FILE, OUTPUT_PDF_FILE)
