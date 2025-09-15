using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.InputSystem; // new input system

public class DoorInteraction : MonoBehaviour
{
    [Tooltip("Name of the scene to load (must be added in Build Settings).")]
    public string sceneToLoad;

    [Tooltip("UI GameObject displaying the 'Press E' prompt.")]
    public GameObject interactPrompt;

    private bool playerInRange = false;
    private PlayerControls controls;

    private void Awake()
    {
        controls = new PlayerControls();
    }

    private void OnEnable()
    {
        controls.Player.Interact.performed += OnInteract;
        controls.Player.Enable();
    }

    private void OnDisable()
    {
        controls.Player.Interact.performed -= OnInteract;
        controls.Player.Disable();
    }

    private void Start()
    {
        if (interactPrompt) interactPrompt.SetActive(false);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            playerInRange = true;
            if (interactPrompt) interactPrompt.SetActive(true);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            playerInRange = false;
            if (interactPrompt) interactPrompt.SetActive(false);
        }
    }

    private void OnInteract(InputAction.CallbackContext context)
    {
        if (playerInRange)
        {
            if (interactPrompt) interactPrompt.SetActive(false);
            SceneManager.LoadScene(sceneToLoad);
        }
    }
}
