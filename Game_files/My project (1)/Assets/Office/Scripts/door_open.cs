using System.Collections;
using UnityEngine;

public class Door : MonoBehaviour, IInteractable
{
    public bool IsOpen = false;
    [SerializeField] private float Speed = 1f;
    [SerializeField] private float RotationAmount = 90f;

    private Vector3 StartRotation;
    private Coroutine AnimationCoroutine;

    private void Awake()
    {
        StartRotation = transform.rotation.eulerAngles;
    }

    public void Interact()
    {
        if (AnimationCoroutine != null)
            StopCoroutine(AnimationCoroutine);

        if (!IsOpen)
        {
            AnimationCoroutine = StartCoroutine(DoRotation(StartRotation, StartRotation + new Vector3(0, RotationAmount, 0)));
            IsOpen = true;
        }
        else
        {
            AnimationCoroutine = StartCoroutine(DoRotation(transform.rotation.eulerAngles, StartRotation));
            IsOpen = false;
        }
    }

    private IEnumerator DoRotation(Vector3 from, Vector3 to)
    {
        Quaternion startRot = Quaternion.Euler(from);
        Quaternion endRot = Quaternion.Euler(to);

        float time = 0;
        while (time < 1)
        {
            transform.rotation = Quaternion.Slerp(startRot, endRot, time);
            yield return null;
            time += Time.deltaTime * Speed;
        }
        transform.rotation = endRot;
    }
}
