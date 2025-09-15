using UnityEngine;
using UnityEngine.SceneManagement;
public class DoorSceneChange : MonoBehaviour,IInteractable
{
    public void Interact()
    {
        SceneManager.LoadSceneAsync(1);
    }

}
