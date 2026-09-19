const get = (id) => document.getElementById(id);
let answers = [];
function element(tag, text) { const e = document.createElement(tag); e.textContent = text; return e; }
function choice(label) {
  const wrapper = element("label", label); const select = document.createElement("select"); select.setAttribute("aria-label", label);
  for (const [value,text] of [["","Choose"],["yes","Yes"],["no","No"],["unclear","Unclear"]]) {
    const option=element("option",text);option.value=value;select.append(option);
  }
  wrapper.append(select);return [wrapper,select];
}
get("packet").onchange = async () => {
  try {
    const file=get("packet").files[0]; if (!file) return;
    if (file.size>10_000_000) throw new Error("Packet exceeds 10 MB.");
    const packet=JSON.parse(await file.text());
    if (!Array.isArray(packet.cases)||!packet.cases.length) throw new Error("No review cases found.");
    answers=[];get("review-cases").replaceChildren();
    for (const c of packet.cases) {
      const section=document.createElement("section");
      section.append(element("h2",c.query),element("p",`References: ${c.reference_answers.join("; ")}`));
      const evidence=document.createElement("details");evidence.append(element("summary","Read full evidence"));
      c.full_candidate_evidence.forEach(d=>evidence.append(element("p",d.text)));section.append(evidence);
      for (const [label,answer] of Object.entries(c.answers)) {
        section.append(element("h3",`Answer ${label}`),element("p",answer.text));
        const [correctLabel,correct]=choice("Semantically correct?");
        const [ambiguousLabel,ambiguous]=choice("Question or reference ambiguous?");
        const notes=element("textarea","");notes.setAttribute("aria-label",`Notes for ${c.case_id} answer ${label}`);
        section.append(correctLabel,ambiguousLabel,notes);
        answers.push({id:c.case_id,label,correct,ambiguous,notes});
      }
      get("review-cases").append(section);
    }
    get("save-review").disabled=false;
    get("review-status").textContent=`${answers.length} answers loaded. No labels have been supplied automatically.`;
  } catch(e) { answers=[];get("save-review").disabled=true;get("review-status").textContent=e.message; }
};
get("save-review").onclick = () => {
  const reviewer=get("reviewer").value.trim();
  if (!reviewer) {get("review-status").textContent="Enter a reviewer identifier.";return;}
  const rows=[["case_id","anonymous_answer","reviewer","semantically_correct","ambiguous_question_or_reference","notes"]];
  for (const a of answers) {
    if ((a.correct.value||a.ambiguous.value)&&(!a.correct.value||!a.ambiguous.value)) {
      get("review-status").textContent="Fill both correctness and ambiguity for each reviewed answer.";return;
    }
    rows.push([a.id,a.label,a.correct.value?reviewer:"",a.correct.value,a.ambiguous.value,a.notes.value]);
  }
  const text=rows.map(row=>row.map(value=>'"'+String(value).replaceAll('"','""')+'"').join(",")).join("\n")+"\n";
  const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([text],{type:"text/csv"}));
  link.download="review-labels.csv";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
  get("review-status").textContent=`Exported ${answers.filter(a=>a.correct.value).length}/${answers.length} labels. Partial review remains incomplete.`;
};
