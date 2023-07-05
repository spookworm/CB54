<table>
    <tbody>
        <tr>
            <td colspan=1>
				<img id="DCUlogo" src="./doc/Dublin_City_University_Logo.png" onerror="this.onerror=null; this.src='file/doc/Dublin_City_University_Logo.png'">
			</td>
            <td colspan=1><h1>CB54: Machine Learning Algorithms for EM Wave Scattering Problems</h1></td>
		</tr>
        <tr>
    </tbody>
</table>
<table>
    <tbody>
            <td rowspan=1>Name</td>
            <td rowspan=1>Anthony James McElwee</td>
        </tr>
        <tr>
            <td rowspan=1>ID Number</td>
            <td rowspan=1>20211330</td>
		</tr>
        <tr>
            <td rowspan=1>Date</td>
            <td rowspan=1>2023 August</td>
		</tr>
        <tr>
            <td colspan=2>MEng in Electronic and Computer Engineering</td>
		</tr>
    </tbody>
</table>


*When an electromagnetic wave encounters an object it scatters, with some energy being transmitted into the object and the rest propagating in a variety of directions depending on the material composition and local geometry. A precise knowledge of the scattering phenomenon is desirable for a variety of applications, such as medical imaging, radar and wireless communications.  Numerical techniques such as the method of moments give highly accurate results, but are computationally expensive. An emerging alternative is the use of machine learning tools that can be trained using a training set of data covering a sufficiently wide feature set (i.e. problem geometry, material, frequency etc). This project will use an in-house, Matlab-based, implementation of the method of moments to train an artificial neural network to solve the problem of EM scattering from convex dielectric bodies.*


<!-- THIS WORKS ON GITHUB: just click on the actual files if using locally -->
[DOWNLOAD Final Project Report](AnthonyJamesMcElwee_20211330_FP.pdf)

[DOWNLOAD IEEE Conference Paper](AnthonyJamesMcElwee_20211330_IEEE_Paper.pdf)

[DOWNLOAD Literature Review](AnthonyJamesMcElwee_20211330_LR_Updated.pdf)

[DOWNLOAD Project Design Plan](AnthonyJamesMcElwee-20211330-PDP-signed.pdf)

[DOWNLOAD Project Research Log](AnthonyJamesMcElwee_20211330_PRL.pdf)

[DOWNLOAD Risk Assignment](AnthonyJamesMcElwee_20211330_RA.pdf)

# Conda Environment Installation
Navigate to the subfolder "env" in conda instance and call the following command:

conda remove --name solveremf2 --all

REM conda create --name solveremf2

conda env create -f solveremf2.yml

# Folder Descriptions
* TBC