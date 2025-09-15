using UnityEngine;

public class CursorLock : MonoBehaviour
{
    void Start()
    {
        Debug.Log("ForceCursorLock.Start() called");
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.L))
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
            Debug.Log("Locked manually");
        }
        if (Input.GetKeyDown(KeyCode.U))
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            Debug.Log("Unlocked manually");
        }
    }
}
