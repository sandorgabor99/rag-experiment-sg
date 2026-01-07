import argparse
import logging
import re
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean text by normalizing line endings, removing page numbers, and fixing formatting."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)

    text = re.sub(r'\n{3,}', '\n\n', text)

    text = re.sub(r'-\n', '', text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description='Clean text files by normalizing formatting and removing artifacts.'
    )
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent 
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(project_root / 'data' / 'init'),
        help='Input directory containing text files (default: clean_txt)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(project_root / 'data' / 'raw'),
        help='Output directory for cleaned text files (default: txt_out)'
    )
    
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    
    # Validate input directory exists
    if not in_dir.exists():
        logger.error(f"Input directory does not exist: {in_dir}")
        sys.exit(1)
    
    if not in_dir.is_dir():
        logger.error(f"Input path is not a directory: {in_dir}")
        sys.exit(1)
    
    # Create output directory
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {out_dir}: {e}")
        sys.exit(1)
    
    # Find all text files
    txt_files = list(in_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"No .txt files found in {in_dir}")
        return
    
    logger.info(f"Found {len(txt_files)} text file(s) to process")
    
    # Process each file
    processed = 0
    failed = 0
    
    for file in txt_files:
        try:
            logger.info(f"Processing: {file.name}")
            raw = file.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw)
            output_file = out_dir / file.name
            output_file.write_text(cleaned, encoding="utf-8")
            processed += 1
            logger.info(f"Successfully processed: {file.name}")
        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
    
    logger.info(f"Processing complete: {processed} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
