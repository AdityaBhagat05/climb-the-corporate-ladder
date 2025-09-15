using UnityEngine;
using UnityEngine.SceneManagement;
public class HangmanBegin : MonoBehaviour,IInteractable
{
    public void Interact()
    {
        Cursor.lockState = CursorLockMode.None;  // Unlock cursor
        Cursor.visible = true; 
        SceneManager.LoadSceneAsync(2);
    }

}
