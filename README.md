# Speaker-Verification-for-Noisy-Data

# Introduction
<ul>
  <li>Noisy environments challenge traditional speaker verification systems.</li>
  <li>That's why we have developed a robust system capable of identifying speakers amidst noise.</li>
  <li>We can mitigate risks associated with impersonization and unaouthorized access, ultimately fortifying the integrity of voice based authentication sytems.</li>
</ul>

### Speaker verification system overview
<p align="center">
<img src="https://github.com/eshajain-25/Speaker-Verification-for-Noisy-Data/assets/114498949/3b3c1d7a-8629-4995-b54c-e26711855231" alt="Overview of speaker verification system" width="400" > 
</p>


## Methodology

<ul>
<li>We have used Resnet-34 pre-trained linear model For getting the embeddings of the audio file which is cleaned in the first step.</li>
  <li>After getting the embeddings we use it to Compare with the embeddings of the ground truth.</li>
  <li>The comparison is based on similarity score. Threshold set for that is 0.5.</li>
</ul>

## Results
<ul>
  <li>Better results were shown when we opted for noise reduction instead of audio enhancement and other filtering techniques</li>
  <li>We observed that there was loss of original information on using too many enhancement and denoising features which resulted in poor results.</li>
  <li>After careful trials on real data and observations a conclusion was made that the threshold for similarity score should be 0.5 .</li>
</ul>

