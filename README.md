<table>
    <tbody>
        <tr>
            <td colspan=1>
				<img id="DCUlogo" src="./media/Dublin_City_University_Logo.png" onerror="this.onerror=null; this.src='file/media/Dublin_City_University_Logo.png'">
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

# Folder Descriptions
* TBC

# Update
* Keep the following documents up-to-date at the end of every session:
	* ./doc/project_management/log_meetings_dates.html
	* ./doc/graphviz_doc.workflow.py

# Notes
* If a python function changes, even an imported function from the custom library, then Gradio will refresh and re-run everything. This may not be wise at the heavier end of the DL process so develop that stage carefully.

# To Do
* Complete transition the final_thesis dropbox folder to this github;
* ~~Put document rendering commands into the project itself for auto update of document links for download. the latest version of the docx and pdf should always be available. since dev activity is not in dropbox a version history of the docx may need to be maintained;~~
* Have workflow render commands itself for auto update of image; (PROBABLY NOT WORTH IT?)
* Have a description file in each folder describing what the folder contents. Auto-populate this readme with those descriptions.
* Need to complete the subdocuments formatting as pages number style etc. is incomplete.

# Active thoughts
aim is to adapt streamlined code so that imported geometry can be used

then that resolution decision is implemented

then move code to python

need to work out the length_x_side from the imported geometry file

floor is not bijective so that means we actually do need to set an import resolution standard

the SI unit for lenght is meter so stick with that? no remember we need resolution at the level that captures geometry features, wave characteristics and can feed into model

so there needs to be a decision between the three which brings the resolution to suitable degree

this may be an iterative decision process.

There are at least three parameters that are required in order to decide on the grid resolution:
* geometry distance scale (0.0, min scale required to capture geometry features];
* model input grid [minimum resolution to allow inference, maximum resolution that memory can handle OR that avoids redundancy to make inference];
* material properties [min is set by floor function & lowest cr & disc sampling & geometry ~~scale~~ length, max is implicitly set by the max disc sampling]

Out of these a grid of x rows and y columns should be generated.

why can't i see the background standing wave in the vacuum? this is the same in the van den berg book. but in other sources it can be seen. is this because of the type of incident wave? do i need to change it so as to be a persisent wave as opposed to an impulse? look at the images in https://www.scielo.br/j/jmoea/a/VvJrMV6dH4rghHZSgg4RpwG/?lang=en

kr(1) is assumed to be vacuum bakground embedding in the code. this is not functional. need to check all of this.

aim is to adapt streamlined code so that imported geometry can be used
then that resolution decision is implemented
then move code to python

need to work out the length_x_side from the imported geometry file
floor is not bijective so that means we actually do need to set an import resolution standard
the SI unit for lenght is meter so stick with that? no remember we need resolution at the level that captures geometry features, wave characteristics and can feed into model

so there needs to be a decision between the three which brings the resolution to suitable degree

RESCALE OBJECT FOR SPECIFIC FREQUENCY: This is required for standardised input into CNN Sophisticated Book uses 128x128. Imported geometry will sit at resolution required to depict physical geometry of object. Then this discretization needs to be checked that it is sufficient to depict the electromagnetic materials of the object. If the discretisation is enough already, then it is maintained. If they discretization needs to be incresed then the imported geometry will be sliced up at a higher resolution. Ultimately, the final resolution of the exported geometry & output field needs to be suitable for model input too.