---
layout: post
title:  "Mitigating cyber flashing in video calls"
date:   2025-01-09 11:53:40 -0800
categories: AI
---
*Content warning: This post discusses **cyber flashing**, including discussion of unsolicited explicit images and experiences related to online sexual harassment. Reader discretion is advised.*

*Note/Disclaimer: I used to work for a online marketplace that used third-party providers to connect people over video chat. These opinions are all my own. All examples of cyber flashing in this article are purely fictional and for the sake of threat analysis.*



My thesis is pretty simple, every video chat provider should allow me to apply a virtual background for everyone in a video call, not just my own video.

Video chat platforms like Google Meet and Zoom assume a high level of trust between meeting members. [Google in particular frames the threat of cyber flashing or disruption as coming from outsiders joining the meeting.](https://support.google.com/meet/answer/9852160?hl=en#zippy=%2Csafety-measures%2Csafety-best-practices) This became well known during the pandemic, apparently becoming known as [Zoombombing](https://en.wikipedia.org/wiki/Zoombombing) (this even triggered an [FBI alert](https://www.fbi.gov/contact-us/field-offices/boston/news/press-releases/fbi-warns-of-teleconferencing-and-online-classroom-hijacking-during-covid-19-pandemic)). Most users are meeting with their coworkers and friends. Cyber flashing in these instances would result in personal consequences that, even if you were unhinged enough to want to do this, make this sort of behavior suicidal and rare. 

On the other hand, Omegle and Chat Roulette were renowned for cyber flashing. They are clear examples of what sort of users are drawn to a platform when there's little to no trust between members. For Salespeople, Recruiters, Online Educators, etc. Google Meet and Zoom might as well be Chat Roulette and Omegle when it comes to trust.    

To their credit, Zoom allows you to turn off video for a call entirely, or disable a particular participant's video after fumbling through a submenu.  Likewise Google [allows you to lock a user's video](https://support.google.com/meet/answer/11275962?hl=en_sg&co=GENIE.Platform%3DDesktop) in the sidebar. Audio-only might be safer but video does a lot to help online communication[^1]. The problem with a simple toggle is that trying to disable video is rather hard to do right after a penis has been burned into your vision.

# Hacking together a browser extension and why do I need to all?

The fact that these options aren't built into video providers frustrates me to no end (though to be fair some providers have third-party add-ons which touch some of these issues). I would provide a link to the code I've written here but I doubt anything I've done is production ready. I believe any extension has critical issues that prevent it from being useful. 

My hope is that either someone tells me I'm wrong and something like this can be made and made well (it would be embarrassing if it already exists) or **puts pressure on major providers to make these feature readily available themselves.** 
I chose a browser extension because a large number of video providers have web clients that _tend_ to use html5 video elements. Theoretically, if you could get this right then content filtering would work well across the entire web and wouldn't require specific integrations with each video provider. 

To start, let's look at detection. Two approaches came to mind, 1) we look for inappropriate content and blur/disable video when we notice it or 2) we blur everything by default and only show "safe content" i.e. we apply a stricter virtual background that only reveals faces.

[^1]: A lot of recruiters do default to phone calls but video provides a human element that matters for things like sales, health, and education.
## Approach 1: hotdog or not
[iykyk](https://www.youtube.com/watch?v=vIci3C4JkL0)

For this, I converted [Bumble's Private Detector](https://github.com/bumble-tech/private-detector/tree/main) to a uint8 quantized version for Tensorflow.js. This drops the model size from ~186 MB to about ~40 MB. 

This is on the higher side of recommended, with the tfjs-converter readme saying,
> While the browser supports loading 100-500MB models, the page load time, the inference time and the user experience would not be great. We recommend using models that are designed for edge devices (e.g. phones). These models are usually smaller than 30MB.

From testing locally and purposefully limiting my CPU using Chrome's dev tools, I was able to achieve about ~10 FPS (where FPS is the number of content checks per second running in the background worker). At this framerate it's likely a user would see _some_ portion of the offensive content but at least we could take action on behalf of people that might freeze up. 

I was actually unable to convert the model inside of a docker image while running on my Mac and thus had to create an instance in AWS to perform the conversion. If anyone really wants the quantized model please email me.

If you want to generate it yourself, here's a dockerfile. Getting the dependencies right turned out to surprisingly quite obnoxious (and in classic ML fashion the readme just says `pip install tensorflowjs` with no versioning information) so here's a dockerfile with pegged dependencies that will get you there yourself:

```docker
FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* 

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv pip install tensorflow==2.4.0 tensorflowjs==2.8.0 --system

ENV PATH="/root/.local/bin:$PATH"

ENTRYPOINT ["/bin/bash"]
```

From there just run the following inside of the image (of course you'll need to mount a path to get the saved image out of the container, or modify the container to upload it to storage somewhere):
```bash
uv python -m tensorflowjs.converters.converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_uint8 \
    /your/model/path \
    /new/model/path
    
```

### Show me a demo!
Um no. Not of this one \(I have a virtual background demo below)! Predictably this strategy involved me trolling around various porn sites trying to find different presentations of cock. [At least I have satire to comfort me](https://youtu.be/dvn-hpZdElo?feature=shared&t=17). Nothing against either porn or cock but not only is this not my thing but browsing porn with such a clinical motive is just off-putting.

I have some comments but I'm not going to go into any specific findings from testing Bumble's model. I don't want to provide any incidental aid to predators. [Instead I'd like to highlight a couple of quotes from one of their devs](https://github.com/bumble-tech/private-detector/issues/17#issuecomment-2654398603),
> The open source PD is not the exact model that we use behind the scenes - we wanted to avoid adversarial testing so we trained a new model, very similar but some slight differences to make sure that we can still keep our users safe

>\[A\] few companies \[...\] said that finetuning the model on a dataset more aligned with their use case makes a much better setup for them

I will say I experienced a high false **negative** rate when using the both the quantized and full models in a video chat context. The publicly available Bumble model, trivially, seems to be tailored to the type of cyber flashing images one might expect to receive on Bumble. **If you do plan to use this model (or any really), like the devs suggest, please finetune it to your specific domain.**

I haven't gotten a chance to try it but there are newer models like [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) which rely on transformers and might dramatically outperform the the 2022 Bumble model. Even still, please do the work and check whether these models work in your domain.  

## Approach 2: virtual backgrounds

### A quick tangent on UX
With approach 1, once we're confident a user is getting flashed and we can take extreme actions. We could block the entire page, we could flat out disable a video element, and if we're worried about false positives we can provide an adjustable blurring filter so the user can figure out what they're looking at without gory detail.

With virtual backgrounds, blurring is happening constantly and if we're running this on any generic html5 video element we want maximum compatibility and to not break the functionality of the page.

In an effort to not modify the DOM too and in part because I am a lowly CRUD webdev, I tried to avoid having to touch a canvas or modify the source of a video element. 

In fact, you can get _so damn close just with CSS_ but there's just two main problems.

#### Video controls exist... 
The first thing I tried was setting `content: blur(20px)` on the video element but to my great annoyance, there is no way to prevent the controls from getting blurred as well.

![Blurring a video element blurs the controls](/assets/images/blur_control_example.png)

#### Exceptions to blur are a pain
The other issue is that the CSS content filter doesn't let you specify an SVG path for applying filters (purely in css). Whereas you can totally specify a clip-path of a canvas with a backdrop filter (think of this as a translucent sheet of glass) and just absolutely position the thing on top of a video element. 

But at this point we're directly modifying the DOM. Meaning, this still blocks both inbuilt controls as well as any custom controls a site might add themselves.

I think you could also modify the immediate structure surrounding the video to wrap it in an SVG element and apply a blur that way. I opted not to try this because modifying styling or adding an outside structure felt safer compared to modifying DOM structure for compatibility sake. 

You could try to exempt the area around the controls but pseudo elements don't report their width/height and in modern UX controls are floating icons over the video and that area might well contain the inappropriate content we're trying to filter out. 

### Using a canvas
So the other option is to hide the original video and then create a canvas we can write frames to (optionally add back yet another video element that reads from the canvas). This would let us do [almost exactly what Google Meet does when applying its virtual backgrounds](https://github.com/tensorflow/tfjs-models/tree/master/body-segmentation#bodysegmentationdrawbokeheffect) (though I don't know if they actually use these models or just provide them).  

Google's demo code just shows the canvas on top of a video element but I don't think that's ideal. We might still want still want default html5 controls for setting the volume or directly pausing the video. If you were to instead have a new video element reading from the canvas, forwarding and syncing the state between two elements video elements is awkward and my shitty implementation resulted in the audio get out of sync from the rendered video frames. 

And, in the end, you're still left at the whims of how each site decides to style and reference the video element. I'm not sure what would work well here, do I hide the video with styling? Should I steal its source and just update the original to read from the canvas?. These both sound bad to me. Hell, Google Meet directly warns you to just write an add-on instead of an extension inside the console logs.

>Developing an extension for Meet? An add-on would work better.
>Extensions frequently cause user issues by altering the page.

## Face Detection and Body Segmentation 
One of the great things about tensorflow.js is that [it comes with a bunch of browser-sized models out of the box](https://www.tensorflow.org/js/models). 

### Face detection
My first thought was face detection. It should be lightweight, and, hell, depending on the domain you can show the video if a head is present on screen and block the video entirely otherwise. After all, one might only expect one head or the other to be visible.

Jokes aside, this trick doesn't work with the video chat domain. Unlike dick picks, people expose themselves with their fully body. I guess predators like subtly? Anyway, we need to blur everything but the face.

I used the [MediaPipeFaceDetector](https://github.com/tensorflow/tfjs-models/tree/master/face-detection) to start. This works reasonably well as you can see below:

<video width="100%" controls muted autoplay loop>
  <source src="/assets/videos/success_face_detection.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


The bounding box helps create a clean cutout that generally will show most of a user's face (with some allowed padding). You'll notice that face detection can cut in and out, so I later implemented a check that requires a percentage of intersecting boxes within one second. This helped protect against false negatives as well as false positives (which isn't dangerous, it's just a bad UX).

Unfortunately the MediaPipeFaceDetector had a critical flaw. Shown below, if a person is relatively small in frame, the bounding box returned by the model ends up being large enough to cover their entire body... which defeats the entire purpose.

<video width="100%" controls muted autoplay loop>
  <source src="/assets/videos/face_detection_size_example.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Thinking back, perhaps I could have recursively called the  model on the image within the bounding box and checked for larger deviations? Anyway, something I'll leave for future work.

### Body Segmentation
Wait, is that an image from their readme blurring _only_ faces?

![Face blur demo from TensorFlow.js](https://github.com/tensorflow/tfjs-models/raw/master/body-segmentation/images/three_people_faceblur.jpg)

It's as if they're mocking me.

Annoyingly, the tfjs library doesn't let you blur both the background and parts of the body at the same time. Running both `drawBokehEffect` and `blurBodyParts` wipes the canvas at the start of the call.

That said, this an easy enough change. I grabbed the code from their [utils file](https://github.com/tensorflow/tfjs-models/blob/18a73420e555a3c1948209ebfa807c7086c7ff0c/shared/calculators/render_util.ts) and change the body mask to only include the face. Easy!



<video width="100%" controls muted autoplay loop>
  <source src="/assets/videos/selfie_comparison.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
<video width="100%" controls muted autoplay loop>
  <source src="/assets/videos/lower_res_comparison.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
<video width="100%" controls muted autoplay loop>
  <source src="/assets/videos/talking_into_laptop_comparison.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

>Q: That looks pretty laggy no?

Yeah well it's not optimized. For high-res stock footage it looks good to me.

The rest of my investigation feels like quite a waste of time. I had assumed body segmentation would only segment the entire body from the background. Little did I know I guess.

Also, as a side note, it turns out the canvas blur filter also averages transparency (which the tfjs library uses as a probability value for whether a pixel is a part of a person or a part of the background) so I needed to write an extra function to wipe transparency before writing everything back out to the canvas (so that it isn't a transparent overlay)

<video width="100%" controls muted autoplay loop>
  <source src="/assets/videos/quick_movement_comparison.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Okay, yeah, it does get pretty slow if there are a lot of people in the image and there's a lot of movement.

## In conclusion
My own shitty code aside, body segmentation works nearly perfectly for mitigating cyber flashing in video calls.

If you are an engineer, at Google, Zoom, etc. please think through low-trust uses of your platforms. This feature shouldn't need to be sketchy or expensive add-on by some third party companies. If you're using a model anything like what Google does this change might be trivial.

Although, the elephant in the room is you'd need to apply this filtering to N videos, not just your own. If performance is an issue, low-data users already get the option to see only the video of whoever is talking, low-trust users could have a similar product experience. Please don't let design awkwardness prevent something like this from launching. Remember that you are sparing countless people from horror.  
