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
<!-- <a href="file/AnthonyJamesMcElwee_20211330_FR.pdf" target="_blank"><h1>DOWNLOAD Final Project Report</h1></a> -->
[DOWNLOAD Final Project Report](AnthonyJamesMcElwee_20211330_FR.pdf)

<!-- <a href="file/AnthonyJamesMcElwee_20211330_PRS.pdf" target="_blank"><h1>DOWNLOAD Project Log Report</h1></a> -->
[DOWNLOAD Project Log Report](AnthonyJamesMcElwee_20211330_PRS.pdf)

# Update
* Keep the following documents up-to-date at the end of every session:
	* ./doc/project_management/project_logs_emailed/
	* ./doc/project_management/AnthonyJamesMcElwee_20211330_PRS.docx
	* ./doc/project_management/log_meetings_dates.html
	* ./doc/graphviz_doc.workflow.py
	* ./AnthonyJamesMcElwee_20211330_FR.docx

# Notes
* If a python function changes, even an imported function from the custom library, then Gradio will refresh and re-run everything. This may not be wise at the heavier end of the DL process so develop that stage carefully.
* DOCX to PDF will only run if the DOCX is newer thant the PDF version.

# To Do
* Transition the final_thesis dropbox folder to this github;
* ~~Put document rendering commands into the project itself for auto update of document links for download. the latest version of the docx and pdf should always be available. since dev activity is not in dropbox a version history of the docx may need to be maintained;~~
* Have workflow render commands itself for auto update of image; (PROBABLY NOT WORTH IT?)