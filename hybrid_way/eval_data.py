"""Evaluation text: 4 topics taught together into ONE memory (interference test),
plus a separate calibration topic used only to pick refusal thresholds.
Each question has a phrase the correct sentence contains. Questions in
"diff" reuse none of the paragraph's key words."""

PARLIAMENT = ("A parliamentary system is a form of government in which the executive branch, the prime minister and cabinet, is drawn from the legislature and depends on its support to stay in power. Voters elect members of parliament, and the party or coalition holding a majority of seats forms the government. The prime minister is the head of government, while a separate head of state, such as a monarch or a president, performs mainly ceremonial duties. Parliament makes laws, approves the budget, and questions ministers. If parliament passes a vote of no confidence, the government must resign or call a new election. This differs from a presidential system, where the president is elected separately and cannot be removed simply by losing a legislative vote. India, the United Kingdom, Canada, and Germany are examples of parliamentary democracies.")
PHOTO = ("Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It takes place mainly in the chloroplasts, which contain the green pigment chlorophyll. The light-dependent reactions capture solar energy and produce ATP and NADPH. The Calvin cycle then uses that chemical energy to build sugar molecules from carbon dioxide. Oxygen is released as a by-product through tiny pores in the leaves called stomata. Photosynthesis is the foundation of most food chains because it produces the energy that other organisms eat.")
SOLAR = ("The solar system consists of the Sun and the objects that orbit it, including eight planets. The four inner planets, Mercury, Venus, Earth, and Mars, are small and rocky. Jupiter and Saturn are gas giants made mostly of hydrogen and helium. Between the orbits of Mars and Jupiter lies the asteroid belt, a region filled with rocky debris. Comets are icy bodies that develop glowing tails when they approach the Sun. The Sun contains more than ninety-nine percent of the total mass of the system.")
WATER = ("The water cycle describes how water moves continuously between the oceans, the atmosphere, and the land. Heat from the Sun causes liquid water to evaporate and rise into the air as vapor. As the vapor cools at higher altitudes, it condenses into tiny droplets that form clouds. When the droplets grow heavy enough, they fall back to the ground as rain, snow, or hail. Some of that water soaks into the soil and becomes groundwater, while the rest runs off into rivers that return it to the sea.")
TOPICS = [PARLIAMENT, PHOTO, SOLAR, WATER]

SAME = [
    ("what happens if parliament passes a vote of no confidence?", "vote of no confidence"),
    ("who forms the government?", "forms the government"),
    ("what does parliament do?", "makes laws"),
    ("which countries are examples of parliamentary democracies?", "Canada"),
    ("who is the head of state?", "ceremonial"),
    ("What is photosynthesis?", "convert sunlight"),
    ("What does the Calvin cycle do?", "Calvin cycle then"),
    ("Where does photosynthesis take place?", "chloroplasts"),
    ("What is the asteroid belt?", "asteroid belt"),
    ("What are comets?", "Comets are icy"),
    ("Which are the inner planets?", "inner planets"),
    ("What is the water cycle?", "water cycle describes"),
    ("What does vapor do when it cools?", "condenses"),
    ("What is groundwater?", "groundwater"),
]
DIFF = [
    ("How can lawmakers force a leader out?", "vote of no confidence"),
    ("Who picks the people that run the country?", "Voters elect"),
    ("Name some nations that rule in that manner?", "Canada"),
    ("What does the assembly do about spending plans?", "approves the budget"),
    ("How is a setup with a directly chosen chief different?", "presidential system"),
    ("Who performs only symbolic roles?", "ceremonial"),
    ("What ends a leader's time in office when lawmakers lose faith in them?", "vote of no confidence"),
    ("How do green plants turn light into food?", "convert sunlight"),
    ("Which part of the cell holds the colouring matter that absorbs light?", "chlorophyll"),
    ("How do leaves let the gas we breathe escape?", "stomata"),
    ("Why do nearly all living things depend on this process for nourishment?", "food chains"),
    ("Where is sugar assembled from air?", "Calvin cycle then"),
    ("What captures the sun's power in the first stage?", "ATP and NADPH"),
    ("Which worlds close to our star are made of stone?", "small and rocky"),
    ("Where is the band of space rubble located?", "asteroid belt"),
    ("What are the giant worlds made of the lightest elements?", "hydrogen and helium"),
    ("Why do frozen wanderers sprout luminous streamers near our star?", "Comets"),
    ("How much of everything in this family of worlds is our star's weight?", "ninety-nine"),
    ("How does warmth from our star turn seas into gas?", "evaporate"),
    ("Why do puffy white formations appear high in the sky?", "form clouds"),
    ("In what forms does moisture drop from the sky?", "rain, snow, or hail"),
    ("What happens to precipitation that sinks into the earth?", "groundwater"),
    ("How does liquid get back to the ocean?", "rivers"),
]
REFUSE = [
    "What is the speed of light?", "How do I bake a chocolate cake?",
    "Who won the football world cup in 2010?", "What is the capital of Australia?",
    "How many bones are in the human hand?", "Which programming language is used for web pages?",
    "What is the boiling point of nitrogen?", "Who wrote Romeo and Juliet?",
]

# ---- calibration topic (never used in the test) ----
DEV_TEXT = ("The immune system defends the body against harmful bacteria, viruses, and other invaders. White blood cells patrol the bloodstream and attack anything that looks foreign. Antibodies are proteins that lock onto specific germs and mark them for destruction. After an infection, memory cells remain and let the body respond faster to the same germ later. Vaccines train the immune system by showing it a harmless piece of a germ.")
DEV_SAME = [("What are antibodies?", "Antibodies are proteins"), ("What does the immune system defend against?", "defends the body")]
DEV_DIFF = [("What protects our organism from microbes?", "defends the body"),
            ("Which cells roam through blood hunting strangers?", "White blood cells"),
            ("Why does a second illness from the same bug hit more gently?", "memory cells"),
            ("How do jabs prepare our defences?", "Vaccines")]
DEV_REFUSE = ["What is the tallest mountain on Earth?", "How does a car engine work?",
              "Who painted the Mona Lisa?", "What is the price of gold today?"]
