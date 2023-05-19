from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get("https://www.selenium.dev/selenium/web/web-form.html")
title = driver.title

emails = [{'text': '\n\nDear Jane Street Team,\n\nI am writing to express my interest in the Software Engineering Internship position for the Winter 2024 Co-op program. As a recent graduate of a computer science program, I am eager to apply my knowledge and skills to a real-world setting.\n\nI am confident that my experience in software engineering, combined with my enthusiasm for problem-solving, would make me an ideal candidate for this position. I am passionate about learning new technologies and have a strong interest in software engineering. I am also a strong communicator and team player, and I am confident that I could contribute to the team in a meaningful way.\n\nI am excited to learn more about the position and the Jane Street team. I am confident that I could be a valuable asset to the team and I look forward to hearing from you.\n\nSincerely,\n\nAli'}, {'text': '\n\nDear Hiring Manager,\n\nMy name is Ali and I am writing to express my interest in the Software Engineering Intern position. After reading the job posting, I am confident that my skills and experience make me an ideal candidate for the role.\n\nI have a strong background in software engineering and have experience working with OCaml, the primary development language used in the position. I am also a highly collaborative individual with a passion for solving interesting problems. I am confident that I can bring a unique perspective to the team and contribute to the success of the organization.\n\nI am excited to learn more about the position and the projects I would be working on. I am confident that I can make a positive impact on the team and am eager to get started.\n\nThank you for your time and consideration.\n\nSincerely,\n\nAli'},
    {'text': '\n\nDear Jane Street,\n\nI am writing to express my interest in the Software Engineering Intern position at Jane Street. I am currently enrolled in a Co-op program at my university and am eager to apply my programming skills to solve interesting problems.\n\nI am a top-notch programmer with a passion for technology and have strong interpersonal and communication skills. I am also fluent in English and am able to work in-person in your NYC office from January 2024 through April 2024.\n\nI am excited to learn more about your interview and team placement processes and to get a sense of what your most recent intern projects looked like. I am confident that I have the skills and enthusiasm to be a valuable asset to your team.\n\nThank you for your time and consideration. I look forward to hearing from you.\n\nSincerely,\n\nAli'}, {'text': '\n\nDear Jane Street Team,\n\nI am writing to express my interest in the Software Engineering Intern position at Jane Street. After researching your company and the work you do, I am excited to apply for this role.\n\nI am currently a student at [University], where I am studying Computer Science. I have a strong background in software engineering, and I am confident that my skills and experience will be an asset to your team. I am particularly interested in the technology-intensive work that Jane Street does, and I am eager to learn more about the systems you build and the tools you use.\n\nI am also excited to learn more about the interview process for this position. I have read the Preparing for a Software Engineering interview at Jane Street page on your website, and I am confident that I have the skills and knowledge to be successful in this role.\n\nI am confident that I can make a positive contribution to your team, and I look forward to hearing from you.\n\nSincerely,\n\nAli'}]

for e in emails:
    print(e['text'])

emails = [{'text': '\n\nDear Hiring Manager,\n\nI am writing to express my interest in the Software Engineering Intern position at AMD. After reading the job posting, I am confident that my skills and experience make me an ideal candidate for the role.\n\nI am a recent graduate of the University of California, Berkeley, with a degree in Computer Science. During my studies, I developed a strong understanding of software engineering principles and have experience working with a variety of programming languages. I am also familiar with embedded firmware and graphics/multimedia drivers.\n\nI am passionate about developing great technology and believe that collaboration, respect, and going the extra mile are essential for achieving unthinkable results. I am eager to join a team that is passionate about disrupting the status quo, pushing boundaries, and delivering innovation.\n\nI am confident that I can make a positive contribution to the team and am excited to learn more about the position. I have attached my resume for your review and would be happy to provide any additional information you may need.\n\nThank you for your time and consideration.\n\nSincerely,\n\nAli'}, {'text': '\n\nDear Hiring Manager,\n\nI am writing to express my interest in the Software Engineering Intern position at your company. After reading the job posting, I am confident that my skills and experience make me an ideal candidate for the role.\n\nI am currently a student at the University of Toronto, studying Computer Science and Engineering. I have a strong proficiency in C and scripting languages such as Bash and Python, and I am familiar with compiler behavior and optimizations (GCC, Clang, ARM). I also have knowledge of Platform Security concepts such as Cryptography, Signing infrastructure, PKI, Secure Boot & AAA concepts, as well as Virtualization concepts.\n\nI am a highly motivated individual who is able to work independently under tight deadlines, responding to changing business and technical conditions with minimal direction. I am confident that I can make a positive contribution to your team and I am excited to learn more about the position.\n\nThank you for your time and consideration. I look forward to hearing from you.\n\nSincerely,\n\nAli'}, {
    'text': '\n\nDear Hiring Manager,\n\nI am writing to express my interest in the Software Engineering Intern position with the Software Security Engineering group. After reading the job posting, I am confident that my skills and experience make me an ideal candidate for this role.\n\nI have a strong background in software engineering, with experience in developing and enabling platform and content security features across the software stack. I am also familiar with AGILE methodologies and have experience in driving cross-team development. I am comfortable working in a fast-paced environment and am confident in my ability to take on new challenges.\n\nI am passionate about writing clean, well-documented code that scales well, and I am confident that I can bring this enthusiasm to the Software Security Engineering group. I am excited to learn more about the position and discuss how I can contribute to the team.\n\nThank you for your time and consideration.\n\nSincerely,\n\nAli'}, {'text': '\n\nDear Hiring Manager,\n\nI am writing to express my interest in the Software Engineering Intern position you recently posted. With my background in cross team development, providing leadership to junior developers, and a passion for writing clean, well-documented code that scales well, I believe I am the perfect candidate for this role.\n\nI am also familiar with the areas you have listed, including embedded platform/firmware development, platform security, SR-IOV/virtualization, secure boot/bootloader/GRUB/BSP development, and kernel development (Linux/FreeBSD/RTOS). I have experience with embedded firmware development on MP architectures such as ARM, MIPS, TI, Freescale, and RISC microprocessors, and I am familiar with embedded concepts such as GPIO, register programming, memory buses, IRQ/FIQ interrupts, instruction pipelining, instruction/data caches, and kernel development concepts. I also have experience with driver/kernel module development.\n\nI am confident that my skills and experience make me an ideal candidate for this position. I am excited to learn more about the role and discuss how I can contribute to the success of your team.\n\nThank you for your time and consideration.\n'}]


for e in emails:
    print(e['text'])
