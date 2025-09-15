using UnityEngine;
public class RandomNumber : MonoBehaviour, IInteractable
{
    public void Interact()
    {
        int randomNumber = Random.Range(1, 100);
        Debug.Log("Generated Random Number: " + randomNumber);
    }
}
