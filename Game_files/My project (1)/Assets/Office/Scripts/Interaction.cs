using UnityEngine;
using UnityEngine.InputSystem;

public interface IInteractable
{
    void Interact();
}

public class Interaction : MonoBehaviour
{
    public Transform InteractiveSource;
    public float InteractionRange = 3f;
    public string interactableTag = "interactable";
    public GameObject interactUIPrefab;

    private GameObject currentUI;

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current.eKey.wasPressedThisFrame)
        {
            Ray r = new Ray(InteractiveSource.position, InteractiveSource.forward);

            if (Physics.Raycast(r, out RaycastHit hitInfo, InteractionRange))
            {
                if (hitInfo.collider.CompareTag(interactableTag))
                {
                    if (hitInfo.collider.gameObject.TryGetComponent(out IInteractable interactObj))
                    {
                        interactObj.Interact();
                    }
                }
            }
        }

        HandleUI();
    }

    void HandleUI()
    {
        Ray r = new Ray(InteractiveSource.position, InteractiveSource.forward);
        if (Physics.Raycast(r, out RaycastHit hitInfo, InteractionRange))
        {
            if (hitInfo.collider.CompareTag(interactableTag))
            {
                if (currentUI == null)
                {
                    currentUI = Instantiate(interactUIPrefab);
                }
                currentUI.transform.position = hitInfo.point + Vector3.up * 0.4f; // slightly above hit point
                currentUI.transform.LookAt(Camera.main.transform);
                currentUI.SetActive(true);
                return;
            }
        }

        if (currentUI != null)
            currentUI.SetActive(false);
    }
}
