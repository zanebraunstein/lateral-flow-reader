def main():
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (720, 1080), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()

    while True:
        frame = picam2.capture_array()

        cv.imshow("Lateral Flow Reader", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break