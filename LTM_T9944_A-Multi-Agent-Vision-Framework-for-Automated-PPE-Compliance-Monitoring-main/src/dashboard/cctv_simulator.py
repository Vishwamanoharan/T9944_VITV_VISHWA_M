
import os
import random
import cv2


class CCTVSimulator:
    """
    Simulates a CCTV camera using the existing PPE dataset images.

    The simulator:
    - Loads images from a directory
    - Cycles through them continuously
    - Resizes frames
    - Optionally applies small camera-like transformations
    - Returns frames for Streamlit display
    """

    def __init__(
        self,
        image_directory="data/images/val",
        width=960,
        height=540,
    ):
        self.image_directory = image_directory
        self.width = width
        self.height = height

        self.images = []
        self.current_index = 0

        self._load_images()

    def _load_images(self):
        """Load all supported image files."""

        if not os.path.exists(self.image_directory):
            raise FileNotFoundError(
                f"Image directory not found: "
                f"{self.image_directory}"
            )

        supported_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        )

        self.images = [
            os.path.join(
                self.image_directory,
                filename,
            )
            for filename in os.listdir(
                self.image_directory
            )
            if filename.lower().endswith(
                supported_extensions
            )
        ]

        self.images.sort()

        if not self.images:
            raise RuntimeError(
                f"No images found in "
                f"{self.image_directory}"
            )

    def reset(self):
        """Restart the simulated CCTV stream."""

        self.current_index = 0

    def get_frame(self):
        """
        Return the next CCTV frame.

        The image sequence loops continuously.
        """

        if not self.images:
            return None

        image_path = self.images[
            self.current_index
        ]

        frame = cv2.imread(image_path)

        if frame is None:
            self.current_index = (
                self.current_index + 1
            ) % len(self.images)

            return self.get_frame()

        frame = cv2.resize(
            frame,
            (self.width, self.height),
        )

        self.current_index = (
            self.current_index + 1
        ) % len(self.images)

        return frame

    def get_frame_with_effect(self):
        """
        Return a frame with a very small simulated
        camera movement.

        This makes the image sequence feel less
        static while remaining lightweight.
        """

        frame = self.get_frame()

        if frame is None:
            return None

        height, width = frame.shape[:2]

        # Small random zoom
        scale = random.uniform(
            1.00,
            1.04,
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        frame = cv2.resize(
            frame,
            (new_width, new_height),
        )

        # Center crop back to dashboard size
        start_x = max(
            0,
            (new_width - width) // 2,
        )

        start_y = max(
            0,
            (new_height - height) // 2,
        )

        frame = frame[
            start_y:start_y + height,
            start_x:start_x + width,
        ]

        # Safety resize
        frame = cv2.resize(
            frame,
            (width, height),
        )

        return frame

    def get_status(self):
        """Return simulator information."""

        return {
            "camera": "CCTV-01",
            "status": "LIVE",
            "total_frames": len(self.images),
            "current_frame": self.current_index,
            "source": self.image_directory,
        }

