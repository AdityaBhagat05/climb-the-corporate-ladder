using UnityEngine;
using UnityEngine.InputSystem; // required for new Input System

public class mouse_look : MonoBehaviour
{
    private float mouseSensitivity = 50f;
    public Transform playerBody;

    float xRotation = 0f;

    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
    }

    void Update()
    {
        // Use Mouse.current.delta for movement
        Vector2 mouseDelta = Mouse.current.delta.ReadValue();

        float mouseX = mouseDelta.x * mouseSensitivity * Time.deltaTime;
        float mouseY = mouseDelta.y * mouseSensitivity * Time.deltaTime;

        // Vertical look (camera only)
        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);
        transform.localRotation = Quaternion.Euler(xRotation, 0f, 0f);

        // Horizontal look (rotate player body)
        playerBody.Rotate(Vector3.up * mouseX);
    }
}
