# Research Project: Does Board Gender Diversity (BGD) Influence Corporate Culture?
# Author: Xinying Fu
# Purpose: This script integrates multiple methodologies to extract corporate culture values from Glassdoor reviews. I want to obtain corporate culture values using Glassdoor dataset as suggested by Abernethy et al. (2021) and Chen (2024) to further test how BGD influences culture 
#    using natural language processing techniques from Abernethy et al. (2021) and Chen (2024), to further investigate the impact of BGD.



# -------------------------- Step 1: Extracting Cultural Values using Stata --------------------------
# This step follows Abernethy et al. (2021) to measure corporate culture values using the Glassdoor dataset.
# Stata is used to generate cultural value scores for Teamwork, Respect, Quality, Integrity, and Innovation.

# Commands in Stata:
# - Generates total word count for pros and cons reviews
# - Counts occurrences of predefined cultural value keywords
# - Normalizes the keyword counts to obtain cultural scores
# - Computes the Abernethy cultural score as an aggregate metric

# Stata script (to be run in Stata):
'''
gen total_words_pros = wordcount(review_pros)
gen total_words_cons = wordcount(review_cons)

* Generate keyword occurrence variables

** Innovation 
gen innovation_count_pros = 0
gen innovation_count_cons = 0

foreach kw in "Innovation" "Creativity" "Innovative" "Innovate" "Innovation" "Creative" "Excellence" "Passion" "World-class" "Technology" "Operational_excellence""Passionate" "Product_innovation" "Capability" "Customer_experience" "Thought_leadership" "Expertise" "Agility" "Efficient" "Technology_innovation" "Competency" "Know-how" "Cutting-edge" "Agile" "Creatively" "Customer-centric" "Enable" "Value_proposition" "Reinvent" "Focus" "Innovation_capability" "Brand" "Technology" "Focus" "Great" "Platform" "Ability" "Best" "Design" "Create" "Solution" "Develop" "Success" "Content" "Capability" "Effort" "Successful" "Efficiency" "Productivity" "Learn" "Unique" "Tool" "Innovation" "Efficient" "Terrific" "Execution" "Exciting" "Enhance" "Business_model" "Enable" "Discipline" {
    replace innovation_count_pros = innovation_count_pros + regexm(review_pros, "`kw'")
    replace innovation_count_cons = innovation_count_cons + regexm(review_cons, "`kw'")
}

** Integrity 
gen integrity_count_pros = 0
gen integrity_count_cons = 0

foreach kw in "Integrity" "Accountability" "Ethic" "Integrity" "Responsibility""Transparency" "Accountable" "Governance" "Ethical" "Transparent" "Trust" "Responsible" "Oversight" "Independence" "Objectivity" "Moral" "Trustworthy" "Fairness" "Hold_accountable" "Corporate_governance" "Autonomy" "Core_value" "Assure" "Stakeholder" "Fiduciary_responsibility" "Continuity" "Credibility" "Honesty" "Privacy" "Fiduciary_duty" "Rigor" "Control" "Management" "Careful" "Honestly" "Regulator" "Honest" "Safety" "Assure" "Compliance" "Trust" "Disciplined" "Responsible" "Proper" "Responsibility" "Thoughtful" "Convince" "Seriously" "Transparent" "Expert" "Consistency" "Candidly" "Transparency" "Responsive" "Truth" "Principle" "Comply" "Board_director" "Thorough" "Conflict" {
    replace integrity_count_pros = integrity_count_pros + regexm(review_pros, "`kw'")
    replace integrity_count_cons = integrity_count_cons + regexm(review_cons, "`kw'")
}

**  Quality 
gen quality_count_pros = 0
gen quality_count_cons = 0

foreach kw in "Quality" "Dedicated" "Quality" "Dedication" "Customer_service" "Customer" "Dedicate" "Service_level" "Mission" "Service_delivery" "Customer_satisfaction" "Service" "Reliability" "Commitment" "Customer_need" "Customer_support" "High-quality" "Ensure" "Customer_relationship" "Quality_service" "Product_quality" "Quality_product" "Capable" "Service_quality" "End_user" "Quality_level" "Customer_expectation" "Service_capability" "Client" "Customer_requirement" "Sla" "Customer" "Product" "Client" "Service" "Build" "Deliver" "Network" "Support" "Quality" "Sales_force" "Infrastructure" "Supplier" "Serve" "Commit" "Field" "Commitment" "Delivery" "Vendor" "Customer_base" "Supply_chain" "Critical" "Requirement" "Ensure" "Speed" "Desire" "Productive" "Gift" "Service_provider" "Capable" "Functionality" {
    replace quality_count_pros = quality_count_pros + regexm(review_pros, "`kw'")
    replace quality_count_cons = quality_count_cons + regexm(review_cons, "`kw'")
}

**  Respect 
gen respect_count_pros = 0
gen respect_count_cons = 0

foreach kw in "Respect" "Talented" "Talent" "Empower" "Team_member" "Employee" "Team" "Leadership" "Leadership_team" "Culture" "Teammate" "Organization" "Entrepreneurial" "Skill" "Executive" "Empowerment" "Management_team" "Best_brightest" "Professionalism" "Staff" "Highly_skilled" "Skill_set" "Technologist" "Competent" "Entrepreneur" "Experienced" "Energize" "Entrepreneurial_spirit" "High-caliber" "Manager" "Leadership_skill" "People" "Team" "Company" "Hire" "Folk" "Organization" "Resource" "Employee" "Management_team" "Train" "Training" "Senior" "Staff" "Member" "Leader" "Person" "Proud" "Talent" "Leadership" "Manager" "CEO" "Knowledge" "Engineer" "Recruit" "Salespeople" "Sales_team" "Consultant" "Culture" "Sales_organization" "Advisor" {
    replace respect_count_pros = respect_count_pros + regexm(review_pros, "`kw'")
    replace respect_count_cons = respect_count_cons + regexm(review_cons, "`kw'")
}

**  Teamwork 
gen teamwork_count_pros = 0
gen teamwork_count_cons = 0

foreach kw in "Teamwork" "Collaborate" "Cooperation" "Collaboration" "Collaborative" "Cooperative" "Partnership" "Cooperate" "Collaboratively" "Partner" "Co-operation" "Coordination" "Engage" "Jointly" "Coordinate" "Teamwork" "Business_partner" "Alliance" "Team_up" "Technology_partner" "Joint" "Cooperatively" "Relationship" "Collaborator" "Interaction" "Working_relationship" "Co-operate" "Technology_partnership" "Association" "Dialogue" "Dialog" "Partner" "Relationship" "Discussion" "Together" "Integrate" "Involve" "Conversation" "Integration" "Partnership" "Engage" "Align" "Explore" "Communication" "Dialogue" "Engagement" "Contact" "Conduct" "On_behalf_of" "Joint" "Collaboration" "Sponsor" "Conjunction" "Supportive" "Alliance" "Merge" "Interaction" "Put_together" "Organize" "Embrace" "Assist" {
    replace teamwork_count_pros = teamwork_count_pros + regexm(review_pros, "`kw'")
    replace teamwork_count_cons = teamwork_count_cons + regexm(review_cons, "`kw'")
}

* Compute normalized scores
gen innovation_score_pros = innovation_count_pros / total_words_pros
gen innovation_score_cons = innovation_count_cons / total_words_cons

gen integrity_score_pros = integrity_count_pros / total_words_pros
gen integrity_score_cons = integrity_count_cons / total_words_cons

gen quality_score_pros = quality_count_pros / total_words_pros
gen quality_score_cons = quality_count_cons / total_words_cons

gen respect_score_pros = respect_count_pros / total_words_pros
gen respect_score_cons = respect_count_cons / total_words_cons

gen teamwork_score_pros = teamwork_count_pros / total_words_pros
gen teamwork_score_cons = teamwork_count_cons / total_words_cons


* Compute aggregate cultural score
gen Abernethy_score_pros = (innovation_score_pros + integrity_score_pros + quality_score_pros + respect_score_pros + teamwork_score_pros) / 5
gen Abernethy_score_cons = (innovation_score_cons + integrity_score_cons + quality_score_cons + respect_score_cons + teamwork_score_cons) / 5

'''

# The output of this Stata step is saved as a CSV file for further analysis in Python.
