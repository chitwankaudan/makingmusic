# makingmusic
Course project for DL 7643

## GAN Approach

Reasons GANs are tough to trains:
[Why GANs are tough to train](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b)
- Non-convergence: the model parameters oscillate, destabilize and never converge
- Mode collapse: the generator collapses which produces limited varieties of samples
- Diminished gradient: the discriminator gets too successful that the generator gradient vanishes and learns nothing

Gan structure:
[GAN architecture for music gen](https://medium.com/ee-460j-final-project/generating-music-with-a-generative-adversarial-network-8d3f68a33096)
- Struggled to train it to not produce random noise. At 250 epochs they found decent results but it didn’t persist. They would like to try an LSTM next

Simple GAN vs. Simple RNN
[Simple GAN for music gen](https://github.com/olofmogren/c-rnn-gan/)
[Simply LSTM for music gen](https://github.com/subpath/Keras_music_gereration/blob/master/Music%20gerenation%20with%20Keras%20and%20TF.ipynb)
- GAN is tougher to train with less impressive results than the simple RNN


