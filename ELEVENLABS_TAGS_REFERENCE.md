# ElevenLabs tags – use directly in your CSV `text` column

These are **ElevenLabs’ built-in parameters**. Write them in the text as-is; no custom implementation needed.  
They work with model **eleven_v3** (script default).

## Emotions
`[happy]` `[sad]` `[excited]` `[calm]` `[nervous]` `[frustrated]` `[sorrowful]` `[angry]` `[tired]`

## Tone
`[cheerfully]` `[flatly]` `[deadpan]` `[playfully]` `[hesitant]` `[regretful]`

## Reactions
`[sigh]` `[laughs]` `[gulps]` `[gasps]` `[whispers]` `[clears throat]` `[laughing]` `[light chuckle]`

## Pacing / cognitive
`[pauses]` `[hesitates]` `[stammers]` `[resigned tone]`

## Volume / delivery
`[whispers]` `[shouts]` `[quietly]` `[loudly]` `[rushed]`

## Example lines for CSV
```text
"[happy] Is this a ball?"
"[pauses] Is this [break] a ball?"
"[sad] I couldn't sleep that night."
"[tired] I've been working 14 hours. [sigh] You sure this will work?"
"speed= low, [question] Is this a ball?"
```

Use **one model** (e.g. `eleven_v3`) so all tags work. If you use `eleven_multilingual_v2`, only SSML (e.g. `<break time="1s" />`) and our `[break]` conversion apply; emotion tags are for v3.
