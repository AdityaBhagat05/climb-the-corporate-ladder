using UnityEngine;
using UnityEngine.InputSystem; // new Input System
public class player_movement : MonoBehaviour
{
    public CharacterController controller;
    private float speed = 5f;

    Vector3 velocity;
    public float gravity = -9.81f;

    public Transform groundCheck;
    public float groundDistance = 0.4f;
    public LayerMask groundMask;
    public float jumpHeight = 2f;
    bool isGrounded;

    void Update()
    {
        isGrounded = Physics.CheckSphere(groundCheck.position, groundDistance, groundMask);
        if (isGrounded && velocity.y < 0)
        {
            velocity.y = -2f; // small negative value to keep grounded
        }

        float x = 0f;
        float z = 0f;

        if (Keyboard.current != null)
        {
            // Horizontal (A/D or Left/Right)
            if (Keyboard.current.aKey.isPressed) x = -1f;
            if (Keyboard.current.dKey.isPressed) x = 1f;

            // Vertical (W/S or Up/Down)
            if (Keyboard.current.wKey.isPressed) z = 1f;
            if (Keyboard.current.sKey.isPressed) z = -1f;
        }

        Vector3 move = transform.right * x + transform.forward * z;

        controller.Move(move * speed * Time.deltaTime);

        if (Keyboard.current.spaceKey.wasPressedThisFrame && isGrounded)
        {
            velocity.y = Mathf.Sqrt(jumpHeight * -gravity); // jump height of 2 units
        }

        velocity.y += gravity * Time.deltaTime;
        
        controller.Move(velocity * Time.deltaTime);
    }
}
