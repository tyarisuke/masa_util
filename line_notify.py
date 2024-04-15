import requests


def send_line_notify_message(token, message):
    """
    LINE Notifyを使用してメッセージを送信する。

    Parameters:
    - token: str, LINE Notify APIトークン
    - message: str, 送信するメッセージ
    """
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    response = requests.post(url, headers=headers, data=data)
    return response.status_code, response.json()


def send_line_notify_image(token, message, image_path):
    """
    LINE Notifyを使用してメッセージと画像を送信する。

    Parameters:
    - token: str, LINE Notify APIトークン
    - message: str, 送信するメッセージ
    - image_path: str, 送信する画像のパス
    """
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    files = {"imageFile": open(image_path, "rb")}
    response = requests.post(url, headers=headers, data=data, files=files)
    return response.status_code, response.json()


if __name__ == "__main__":
    # 使用例
    token = "YOUR_LINE_NOTIFY_TOKEN"
    message = "こんにちは、これはテストメッセージです。"
    image_path = "/path/to/your/image.jpg"

    # メッセージを送信
    status_code, response_json = send_line_notify_message(token, message)
    print(status_code, response_json)

    # メッセージと画像を送信
    status_code, response_json = send_line_notify_image(
        token, message, image_path
    )
    print(status_code, response_json)
