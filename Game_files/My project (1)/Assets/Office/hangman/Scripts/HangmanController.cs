using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HangmanController : MonoBehaviour
{
    [SerializeField] GameObject wordContainer;
    [SerializeField] GameObject keyboardContainer;
    [SerializeField] GameObject letterContainer;
    [SerializeField] GameObject[] hangmanStages;
    [SerializeField] GameObject letterButton;
    [SerializeField] TextAsset possibleWord;

    private string word;
    private int incorrectGuesses, correctGuesses;

    void Start()
    {
        InitialiseButtons();
        InitialiseGame();
    }

    private void InitialiseButtons()
    {
        for (int i = 65; i <= 90; i++)
        {
            CreateButton(i);
        }
    }

    private void InitialiseGame()
    {
        incorrectGuesses = 0;
        correctGuesses = 0;

        foreach (Button child in keyboardContainer.GetComponentsInChildren<Button>())
        {
            child.interactable = true;
        }

        // Destroy only children, not the container itself
        foreach (Transform child in wordContainer.transform)
        {
            Destroy(child.gameObject);
        }

        foreach (GameObject stage in hangmanStages)
        {
            stage.SetActive(false);
        }

        word = generateWord().ToUpper();
        foreach (char letter in word)
        {
            Instantiate(letterContainer, wordContainer.transform);
        }
    }

    private void CreateButton(int i)
    {
        GameObject temp = Instantiate(letterButton, keyboardContainer.transform);

        TextMeshProUGUI tmp = temp.GetComponentInChildren<TextMeshProUGUI>();
        if (tmp == null)
        {
            Debug.LogError("LetterButton prefab is missing a TextMeshProUGUI child!");
            return;
        }
        tmp.text = ((char)i).ToString();

        Button btn = temp.GetComponent<Button>();
        if (btn == null)
        {
            Debug.LogError("LetterButton prefab is missing a Button component!");
            return;
        }

        btn.onClick.AddListener(() =>
        {
            CheckLetter(((char)i).ToString());
            btn.interactable = false;
        });
    }

    private string generateWord()
    {
        string[] wordsList = possibleWord.text.Split('\n');
        string line = wordsList[Random.Range(0, wordsList.Length - 1)];
        return line.Trim();
    }

    private void CheckLetter(string inputLetter)
    {
        bool letterInWord = false;
        for (int i = 0; i < word.Length; i++)
        {
            if (inputLetter == word[i].ToString())
            {
                letterInWord = true;
                correctGuesses++;
                wordContainer.GetComponentsInChildren<TextMeshProUGUI>()[i].text = inputLetter;
            }
        }
        if (!letterInWord)
        {
            incorrectGuesses++;
            hangmanStages[incorrectGuesses - 1].SetActive(true);
        }
        CheckOutcome();
    }

    private void CheckOutcome()
    {
        if (correctGuesses == word.Length)
        {
            for (int i = 0; i < word.Length; i++)
            {
                wordContainer.GetComponentsInChildren<TextMeshProUGUI>()[i].color = Color.green;
            }
            Invoke("InitialiseGame", 3f);
        }
        if (incorrectGuesses == hangmanStages.Length)
        {
            for (int i = 0; i < word.Length; i++)
            {
                wordContainer.GetComponentsInChildren<TextMeshProUGUI>()[i].color = Color.red;
                wordContainer.GetComponentsInChildren<TextMeshProUGUI>()[i].text = word[i].ToString();
            }
            Invoke("InitialiseGame", 3f);
        }
    }

    void Update()
    {
        
    }
}
